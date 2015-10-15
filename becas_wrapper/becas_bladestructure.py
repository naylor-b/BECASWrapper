
import numpy as np
import time
import os
from collections import OrderedDict
from scipy.interpolate import pchip

from openmdao.core import Component, Group, ParallelGroup
from openmdao.components import IndepVarComp

from cs2dtobecas import CS2DtoBECAS
from becas_wrapper import BECASWrapper


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

    outputs
    -------
    cs_props: array
        vector of cross section properties. Size (19) or (30)
        for standard HAWC2 output or the fully populated stiffness
        matrix, respectively.
    """

    def __init__(self, config, st3d, s, ni_chord, cs_size):
        super(BECASCSStructure, self).__init__()

        self.basedir = os.getcwd()

        self.nr = len(st3d['regions'])
        self.ni_chord = ni_chord

        # add materials properties array ((10, nmat))
        self.add_param('matprops', st3d['matprops'])

        # add materials strength properties array ((18, nmat))
        self.add_param('failmat', st3d['failmat'])

        # add DPs array
        self.add_param('DPs', np.zeros(self.nr + 1))

        # add coords coords
        self.add_param('coords', np.zeros((ni_chord, 3)))

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
                varname = 'r%02d%s' % (ireg, lname)
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
                varname = 'w%02d%s' % (ireg, lname)
                self.add_param(varname + 'T', 0.)
                self.add_param(varname + 'A', 0.)


        # add outputs
        self.add_output('cs_props', np.zeros(cs_size))

        self.mesher = CS2DtoBECAS(self.cs2d, **config['CS2DtoBECAS'])
        self.becas = BECASWrapper(self.cs2d['s'], **config['BECASWrapper'])

    def params2dict(self, params):

        self.cs2d['coords'] = params['coords'][:, :2]
        self.cs2d['matprops'] = params['matprops']
        self.cs2d['failmat'] = params['failmat']
        self.cs2d['DPs'] = params['DPs']
        for ireg, reg in enumerate(self.cs2d['regions']):
            Ts = []
            As = []
            layers = []
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
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
                varname = 'w%02d%s' % (ireg, lname)
                if params[varname + 'T'] > 0.:
                    Ts.append(params[varname + 'T'])
                    As.append(params[varname + 'A'])
                    layers.append(lname)
            self.cs2d['webs'][ireg]['thicknesses'] = np.asarray(Ts)
            self.cs2d['webs'][ireg]['angles'] = np.asarray(As)
            self.cs2d['webs'][ireg]['layers'] = layers

    def solve_nonlinear(self, params, unknowns, resids):

        if self.comm:
            print 'rank %i computing props for section %3.3f %i %i' % \
                                    (self.comm.rank, params['coords'][0, 2],
                                     params['coords'].shape[0], params['coords'].shape[1])
        else:
            print 'computing props for section', params['coords'][0, 2]

        workdir = 'sec%3.3f' % self.cs2d['s']
        try:
            os.mkdir(workdir)
        except:
            pass
        os.chdir(workdir)

        self.params2dict(params)

        self.mesher.cs2d = self.cs2d
        self.mesher.compute()
        self.becas.compute()
        self.unknowns['cs_props'] = self.becas.cs_props

        os.chdir(self.basedir)


class Slice(Component):
    """
    simple component for slicing arrays into vectors
    for passing to sub-comps computing the csprops

    parameters
    ----------
    DPs: array
        2D array of DPs. Size: ((nsec, nDP))
    surface: array
        blade surface. Size: ((ni_chord, nsec, 3))

    outputs
    -------
    sec<xxx>DPs: array
        Vector of DPs along chord for each section. Size (nDP)
    sec<xxx>coords: array
        Array of cross section coords shapes. Size ((ni_chord, 3))
    """

    def __init__(self, DPs, surface):
        super(Slice, self).__init__()

        self.nsec = surface.shape[1]

        self.add_param('DPs', DPs)
        self.add_param('surface', surface)
        for i in range(self.nsec):
            self.add_output('sec%03dDPs' % i, DPs[i, :])
            self.add_output('sec%03dcoords' % i, surface[:, i, :])

    def solve_nonlinear(self, params, unknowns, resids):

        for i in range(self.nsec):
            unknowns['sec%03dDPs' % i] = params['DPs'][i, :]
            unknowns['sec%03dcoords' % i] = params['surface'][:, i, :]


class Postprocess(Component):
    """
    component for gathering cross section props
    into array as function of span

    parameters
    ----------
    cs_props<xxx>: array
        array of cross section props. Size (19).
    blade_s: array
        dimensionalised running length of blade
    hub_radius: float
        dimensionalised hub radius

    outputs
    -------
    beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    """

    def __init__(self, nsec, cs_size):
        super(Postprocess, self).__init__()

        self.nsec = nsec

        for i in range(nsec):
            self.add_param('cs_props%03d' % i, np.zeros(cs_size))
        self.add_param('blade_s', np.zeros(nsec))
        self.add_param('hub_radius', 0.)


        self.add_output('beam_structure', np.zeros((nsec, cs_size)))
        self.add_output('blade_mass', 0.)
        self.add_output('blade_mass_moment', 0.)

    def solve_nonlinear(self, params, unknowns, resids):

        for i in range(self.nsec):
            cname = 'cs_props%03d' % i
            cs = params[cname]
            unknowns['beam_structure'][i, :] = cs


class BECASBeamStructure(Group):
    """
    Group for computing beam structure properties
    using the cross-sectional structure code BECAS.

    The geometric and structural inputs used are defined
    in detail in FUSED-Wind.

    parameters
    ----------
    matprops: array
        material stiffness properties. Size (10, nmat).
    failmat: array
        material strength properties. Size (18, nmat).
    DPs: array
        2D array of DPs. Size: ((nsec, nDP))
    surface: array
        blade surface. Size: ((ni_chord, nsec, 3))
    r<xx><lname>T: array
        region layer thicknesses, e.g. r01triaxT. Size (nsec)
    r<xx><lname>A: array
        region layer angles, e.g. r01triaxA. Size (nsec)
    w<xx><lname>T: array
        web layer thicknesses, e.g. r01triaxT. Size (nsec)
    w<xx><lname>A: array
        web layer angles, e.g. r01triaxA. Size (nsec)

    outputs
    -------
    beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    blade_mass: float
        blade mass integrated from beam_structure dm
    """

    def __init__(self, config, st3d, surface):
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
            blade surface. Size: ((ni_chord, nsec, 3))
        """
        super(BECASBeamStructure, self).__init__()

        try:
            if config['BECASCSStructure']['BECASWrapper']['hawc2_FPM']:
                cs_size = 30
            else:
                cs_size = 19
        except:
            cs_size = 19

        self.st3d = st3d
        nr = len(st3d['regions'])
        nsec = st3d['s'].shape[0]

        # add materials properties array ((10, nmat))
        self.add('matprops_c', IndepVarComp('matprops', st3d['matprops']), promotes=['*'])

        # add materials strength properties array ((18, nmat))
        self.add('failmat_c', IndepVarComp('failmat', st3d['failmat']), promotes=['*'])

        # add DPs array with s, DP0, DP1, ... DP<nr>
        self.add('DPs_c', IndepVarComp('DPs', st3d['DPs']), promotes=['*'])

        # add array containing blade section coords
        self.add('surface_c', IndepVarComp('surface', surface), promotes=['*'])

        # add comp to slice the 2D arrays DPs and surface
        self.add('slice', Slice(st3d['DPs'], surface), promotes=['*'])

        self._varnames = []
        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                self.add(varname+'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                self.add(varname+'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])
                self._varnames.append(varname)
        for ireg, reg in enumerate(st3d['webs']):
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                self.add(varname+'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                self.add(varname+'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])
                self._varnames.append(varname)

        # now add a component for each section
        cid = self.add('cid', ParallelGroup())

        for i in range(nsec):
            secname = 'sec%03d' % i
            cid.add(secname, BECASCSStructure(config['BECASCSStructure'], st3d,
                                              st3d['s'][i], surface.shape[0], cs_size))
            # create connections
            self.connect('matprops', 'cid.%s.matprops' % secname)
            self.connect('failmat', 'cid.%s.failmat' % secname)
            self.connect(secname+'DPs', 'cid.%s.DPs' % secname)
            self.connect(secname+'coords', 'cid.%s.coords' % secname)

            for name in self._varnames:
                self.connect(name + 'T', 'cid.%s.%sT' % (secname, name), src_indices=([i]))
                self.connect(name + 'A', 'cid.%s.%sA' % (secname, name), src_indices=([i]))

        self.add('postpro', Postprocess(nsec, cs_size), promotes=['hub_radius',
                                                         'blade_s',
                                                         'beam_structure',
                                                         'blade_mass',
                                                         'blade_mass_moment'])
        for i in range(nsec):
            secname = 'sec%03d' % i
            self.connect('cid.%s.cs_props' % secname, 'postpro.cs_props%03d' % i)
