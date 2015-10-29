
import numpy as np
import time
import os
from collections import OrderedDict
from scipy.interpolate import pchip

from openmdao.api import Component, Group, ParallelGroup
from openmdao.api import IndepVarComp

from cs2dtobecas import CS2DtoBECAS
from becas_wrapper import BECASWrapper

from fusedwind.lib.geom_tools import calculate_length

class BECASCSStructure(Component):
    """
    Component for computing beam structural properties
    using the cross-sectional structure code BECAS.

    The code firstly calls CS2DtoBECAS which is a wrapper around
    shellexpander that comes with BECAS, and second
    calls BECAS using a file interface.

    parameters
    ----------
    config: dict
        dictionary of model specific inputs
    coords: array
        cross-sectional shape. size ((ni_chord, 3))
    matprops: array
        material stiffness properties. Size ((10, nmat)).
    failmat: array
        material strength properties. Size ((18, nmat)).
    DPs: array
        vector of DPs. Size: (nDP)
    coords: array
        blade section coordinates. Size: ((ni_chord, 3))
    r<xx><lname>T: float
        layer thicknesses, e.g. r01triaxT.
    r<xx><lname>A: float
        layer angles, e.g. r01triaxA.
    w<xx><lname>T: float
        web thicknesses, e.g. r01triaxT.
    w<xx><lname>A: float
        web angles, e.g. r01triaxA.

    returns
    -------
    cs_props: array
        vector of cross section properties. Size (19) or (30)
        for standard HAWC2 output or the fully populated stiffness
        matrix, respectively.
    """

    def __init__(self, name, config, st3d, s, ni_chord, cs_size):
        """
        parameters
        ----------
        config: dict
            dictionary with inputs to CS2DtoBECAS and BECASWrapper
        st3d: dict
            dictionary with blade structural definition
        s: array
            spanwise location of the cross-section
        ni_chord: int
            number of points definiting the cross-section shape
        cs_size: int
            size of blade_beam_structure array (19 or 30)
        """
        super(BECASCSStructure, self).__init__()

        self.basedir = os.getcwd()

        self.nr = len(st3d['regions'])
        self.ni_chord = ni_chord

        # add materials properties array ((10, nmat))
        self.add_param('matprops', st3d['matprops'])

        # add materials strength properties array ((18, nmat))
        self.add_param('failmat', st3d['failmat'])

        # add DPs array
        self.add_param('%s:DPs' % name, np.zeros(self.nr + 1))

        # add coords coords
        self.add_param('%s:coords' % name, np.zeros((ni_chord, 3)))

        self.cs2d = {}
        self.cs2d['materials'] = st3d['materials']
        self.cs2d['matprops'] = st3d['matprops']
        self.cs2d['failcrit'] = st3d['failcrit']
        self.cs2d['failmat'] = st3d['failmat']
        self.cs2d['web_def'] = st3d['web_def']
        self.cs2d['s'] = s
        self.cs2d['DPs'] = np.zeros(self.nr + 1)
        self.cs2d['regions'] = []
        self.cs2d['webs'] = []
        for ireg, reg in enumerate(st3d['regions']):
            r = {}
            r['layers'] = reg['layers']
            nl = len(reg['layers'])
            r['thicknesses'] = np.zeros(nl)
            r['angles'] = np.zeros(nl)
            self.cs2d['regions'].append(r)
            for i, lname in enumerate(reg['layers']):
                varname = '%s:r%02d%s' % (name, ireg, lname)
                self.add_param(varname + 'T', 0.)
                self.add_param(varname + 'A', 0.)
        for ireg, reg in enumerate(st3d['webs']):
            r = {}
            r['layers'] = reg['layers']
            nl = len(reg['layers'])
            r['thicknesses'] = np.zeros(nl)
            r['angles'] = np.zeros(nl)
            self.cs2d['webs'].append(r)
            for i, lname in enumerate(reg['layers']):
                varname = '%s:w%02d%s' % (name, ireg, lname)
                self.add_param(varname + 'T', 0.)
                self.add_param(varname + 'A', 0.)


        # add outputs
        self.add_output('%s:cs_props' % name, np.zeros(cs_size))

        self.mesher = CS2DtoBECAS(self.cs2d, **config['CS2DtoBECAS'])
        self.becas = BECASWrapper(self.cs2d['s'], **config['BECASWrapper'])

    def _params2dict(self, params):
        """
        convert the OpenMDAO params dictionary into
        the dictionary format used in CS2DtoBECAS.
        """

        self.cs2d['coords'] = params['%s:coords' % self.name][:, :2]
        print self.name, params['%s:coords' % self.name][:, :2].shape
        self.cs2d['matprops'] = params['matprops']
        self.cs2d['failmat'] = params['failmat']
        self.cs2d['DPs'] = params['%s:DPs' % self.name]
        for ireg, reg in enumerate(self.cs2d['regions']):
            Ts = []
            As = []
            layers = []
            for i, lname in enumerate(reg['layers']):
                varname = '%s:r%02d%s' % (self.name, ireg, lname)
                if params[varname + 'T'] > 0.:
                    Ts.append(params[varname + 'T'])
                    As.append(params[varname + 'A'])
                    layers.append(lname)
            self.cs2d['regions'][ireg]['thicknesses'] = np.asarray(Ts)
            self.cs2d['regions'][ireg]['angles'] = np.asarray(As)
            self.cs2d['regions'][ireg]['layers'] = layers
        for ireg, reg in enumerate(self.cs2d['webs']):
            Ts = []
            As = []
            layers = []
            for i, lname in enumerate(reg['layers']):
                varname = '%s:w%02d%s' % (self.name, ireg, lname)
                if params[varname + 'T'] > 0.:
                    Ts.append(params[varname + 'T'])
                    As.append(params[varname + 'A'])
                    layers.append(lname)
            self.cs2d['webs'][ireg]['thicknesses'] = np.asarray(Ts)
            self.cs2d['webs'][ireg]['angles'] = np.asarray(As)
            self.cs2d['webs'][ireg]['layers'] = layers

    def solve_nonlinear(self, params, unknowns, resids):
        """
        calls CS2DtoBECAS/shellexpander to generate mesh
        and BECAS to compute the cs_props
        """
        workdir = 'sec%3.3f' % self.cs2d['s']
        try:
            os.mkdir(workdir)
        except:
            pass
        os.chdir(workdir)

        self._params2dict(params)

        self.mesher.cs2d = self.cs2d
        self.mesher.compute()
        self.becas.compute()
        self.unknowns['%s:cs_props' % self.name] = self.becas.cs_props

        os.chdir(self.basedir)


class Slice(Component):
    """
    simple component for slicing arrays into vectors
    for passing to sub-comps computing the csprops

    parameters
    ----------
    DP<xx>: array
        arrays of DPs along span. Size: (nsec)
    blade_surface_norm_st: array
        blade surface. Size: ((ni_chord, nsec, 3))

    returns
    -------
    sec<xxx>DPs: array
        Vector of DPs along chord for each section. Size (nDP)
    sec<xxx>coords: array
        Array of cross section coords shapes. Size ((ni_chord, 3))
    """

    def __init__(self, st3d, sdim):
        """
        parameters
        ----------
        DPs: array
            DPs array, size: ((nsec, nDP))
        sdim: array
            blade surface. Size: ((ni_chord, nsec, 3))
        """
        super(Slice, self).__init__()

        self.nsec = sdim[1]
        DPs = st3d['DPs']
        self.nDP = DPs.shape[1]

        for i in range(self.nDP):
            self.add_param('DP%02d' % i, DPs[:, i])


        self.add_param('blade_surface_st', np.zeros(sdim))

        for i in range(self.nsec):
            self.add_output('sec%03d:DPs' % i, DPs[i, :])
            self.add_output('sec%03d:coords' % i, np.zeros((sdim[0], sdim[2])))

    def solve_nonlinear(self, params, unknowns, resids):

        for i in range(self.nsec):
            DPs = np.zeros(self.nDP)
            for j in range(self.nDP):
                DPs[j] = params['DP%02d' % j][i]
            unknowns['sec%03d:DPs' % i] = DPs
            unknowns['sec%03d:coords' % i] = params['blade_surface_st'][:, i, :]


class PostprocessCS(Component):
    """
    component for gathering cross section props
    into array as function of span

    parameters
    ----------
    cs_props<xxx>: array
        array of cross section props. Size (19).
    blade_x: array
        dimensionalised x-coordinate of blade axis
    blade_y: array
        dimensionalised y-coordinate of blade axis
    blade_z: array
        dimensionalised z-coordinate of blade axis
    hub_radius: float
        dimensionalised hub length

    returns
    -------
    blade_beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    blade_mass: float
        blade mass integrated from dm in beam properties
    blade_mass_moment: float
        blade mass moment integrated from dm in beam properties
    """

    def __init__(self, nsec, cs_size):
        """
        parameters
        ----------
        nsec: int
            number of blade sections.
        cs_size: int
            size of blade_beam_structure array (19 or 30).
        """
        super(PostprocessCS, self).__init__()

        self.nsec = nsec

        for i in range(nsec):
            self.add_param('cs_props%03d' % i, np.zeros(cs_size), desc='cross-sectional props for sec%03d' % i)
        self.add_param('hub_radius', 0., units='m', desc='Hub length')
        self.add_param('blade_length', 0., units='m', desc='Blade length')

        self.add_param('x_st', np.zeros(nsec), units='m', desc='dimensionalised x-coordinate of blade axis')
        self.add_param('y_st', np.zeros(nsec), units='m', desc='dimensionalised y-coordinate of blade axis')
        self.add_param('z_st', np.zeros(nsec), units='m', desc='dimensionalised y-coordinate of blade axis')


        self.add_output('blade_beam_structure', np.zeros((nsec, cs_size)), desc='Beam properties of the blade')
        self.add_output('blade_mass', 0., units='kg', desc='Blade mass')
        self.add_output('blade_mass_moment', 0., units='N*m', desc='Blade mass moment')

    def solve_nonlinear(self, params, unknowns, resids):
        """
        aggregate results and integrate mass and mass moment using np.trapz.
        """
        for i in range(self.nsec):
            cname = 'cs_props%03d' % i
            cs = params[cname]
            unknowns['blade_beam_structure'][i, :] = cs

        # compute mass and mass moment
        x = params['x_st'] * params['blade_length']
        y = params['y_st'] * params['blade_length']
        z = params['z_st'] * params['blade_length']
        hub_radius = params['hub_radius']
        s = calculate_length(np.array([x, y, z]).T)
        dm = unknowns['blade_beam_structure'][:, 1]
        g = 9.81

        # mass
        m = np.trapz(dm, s)
        unknowns['blade_mass'] = m

        # mass moment
        mm = np.trapz(g * dm * (z + hub_radius), s)
        unknowns['blade_mass_moment'] = mm

        print('blade mass %10.3f' % m)


class BECASBeamStructure(Group):
    """
    Group for computing beam structure properties
    using the cross-sectional structure code BECAS.

    The geometric and structural inputs used are defined
    in detail in FUSED-Wind.

    parameters
    ----------
    blade_x: array
        dimensionalised x-coordinates of blade axis with structural discretization.
    blade_y: array
        dimensionalised y-coordinates of blade axis with structural discretization.
    blade_z: array
        dimensionalised z-coordinates of blade axis with structural discretization.
    blade_surface_st: array
        blade surface with structural discretization. Size: ((ni_chord, nsec, 3))
    matprops: array
        material stiffness properties. Size (10, nmat).
    failmat: array
        material strength properties. Size (18, nmat).
    sec<xx>DPs: array
        2D array of DPs. Size: ((nsec, nDP))
    sec<xx>coords: array
        blade surface. Size: ((ni_chord, nsec, 3))
    sec<xx>r<yy><lname>T: array
        region layer thicknesses, e.g. r01triaxT. Size (nsec)
    sec<xx>r<yy><lname>A: array
        region layer angles, e.g. r01triaxA. Size (nsec)
    sec<xx>w<yy><lname>T: array
        web layer thicknesses, e.g. r01triaxT. Size (nsec)
    sec<xx>w<yy><lname>A: array
        web layer angles, e.g. r01triaxA. Size (nsec)

    returns
    -------
    blade_beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    blade_mass: float
        blade mass integrated from blade_beam_structure dm
    blade_mass_moment: float
        blade mass moment integrated from blade_beam_structure dm
    """

    def __init__(self, group, config, st3d, sdim):
        """
        initializes parameters and adds a csprops component
        for each section

        parameters
        ----------
        config: dict
            dictionary of inputs for the cs_code class
        st3d: dict
            dictionary of blade structure properties
        surface: array
            blade surface with structural discretization.
            Size: ((ni_chord, nsec, 3))
        """
        super(BECASBeamStructure, self).__init__()

        # check that the config is ok
        if not 'CS2DtoBECAS' in config.keys():
            raise RuntimeError('You need to supply a config dict',
                               'for CS2DtoBECAS')
        if not 'BECASWrapper' in config.keys():
            raise RuntimeError('You need to supply a config dict',
                               'for BECASWrapper')
        try:
            analysis_mode = config['BECASWrapper']['analysis_mode']
            if not analysis_mode == 'stiffness':
                config['BECASWrapper']['analysis_mode'] = 'stiffness'
                print 'BECAS analysis mode wasnt set to `stiffness`,',\
                      'trying to set it for you'
        except:
            print 'BECAS analysis mode wasnt set to `stiffness`,',\
                  'trying to set it for you'
            config['BECASWrapper']['analysis_mode'] = 'stiffness'

        try:
            if config['BECASWrapper']['hawc2_FPM']:
                cs_size = 30
            else:
                cs_size = 19
        except:
            cs_size = 19

        self.st3d = st3d
        nr = len(st3d['regions'])
        nsec = st3d['s'].shape[0]

        # add comp to slice the 2D arrays DPs and surface
        self.add('slice', Slice(st3d, sdim), promotes=['*'])

        self._varnames = []
        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                self._varnames.append(varname)
        for ireg, reg in enumerate(st3d['webs']):
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                self._varnames.append(varname)

        # # now add a component for each section
        par = self.add('par', ParallelGroup(), promotes=['*'])

        for i in range(nsec):
            secname = 'sec%03d' % i
            par.add(secname, BECASCSStructure(secname, config, st3d,
                                              st3d['s'][i], sdim[0], cs_size), promotes=['*'])

            # passing the parent group down is not nice,
            # but makes things a lot cleaner
            for name in self._varnames:
                group.connect(name + 'T', '%s:%sT' % (secname, name), src_indices=([i]))
                group.connect(name + 'A', '%s:%sA' % (secname, name), src_indices=([i]))

        promotions = ['hub_radius',
                      'blade_length',
                      'x_st',
                      'y_st',
                      'z_st',
                      'blade_beam_structure',
                      'blade_mass',
                      'blade_mass_moment']
        self.add('postpro', PostprocessCS(nsec, cs_size), promotes=promotions)
        for i in range(nsec):
            secname = 'sec%03d' % i
            self.connect('%s:cs_props' % secname, 'postpro.cs_props%03d' % i)
