
__all__ = ['BECASWrapper']

import os
import numpy as np
import time
import copy
import commands

def ksfunc(p, rho=50., side=1.):
    """
    Kreisselmeier and Steinhauser constraint aggregation function

    params
    --------
    p: 1-d array
         vector of constraints
    beta: float
        aggregation parameter
    side: float
        side=1 computes the max, side=-1. computes the min
    """
    p = p.flatten()
    side = np.float(side)
    pmax = p.max()

    return pmax + side * np.log(np.sum(np.exp(side * rho * (p - pmax)))) / rho


class BECASWrapper(object):
    u"""
    Python wrapper for BECAS 2D
    =============================

    A basic wrapper of BECAS that provides two primary functionalities:
    computing cross-sectional properties and computing stresses,
    strains and check for material failure.

    parameters
    ----------
    exec_mode: str
        options: 'oct2py', 'octave', 'matlab'.
        Run BECAS either using the Oct2Py bridge, a system call to matlab or to octave.
    analysis_mode: str
        options: 'stiffness', 'stress_recovery', 'combined'.
        call BECAS to either compute stiffness properties
        or to recover stresses or both.
    utils_rst_filebase: str
        file base name for mat files saved with BECAS utils. Default 'becas_utils'.
    path_becas: str
        absolute path to BECAS source files
    timeout: float
        timeout of BECAS call (only used in Oct2Py mode)
    path_input: str
        Relative path BECAS input files
    path_plots: str
        relative path to directory where plots are saved
    checkmesh: bool
        Activate BECAS check mesh
    plot_paraview: bool
        Export plot files for ParaView
    span_pos: float
        spanwise position of section along blade
    hawc2_FPM: bool
        Compute fully populated stiffness matrix beam properties
    rho_ks: float
        Kreisselmeier and Steinhauser aggregation parameter
    nl_2d : Array, default=None
        (:math:`n_n*3`) array with the list of nodal positions where each row
        is in the form (node number, x coordinate, y coordinate), where n_n is
        the total number of nodes. The node numbering need not be in any
        specific order.
    el_2d : array, default=None
        (:math:`n_e*8`) array with the element connectivity table where each
        row is in the form (element number, node 1, node 2, node 3, node 4,
        node 5, node 6, node 7, node 8), where n e is the total number of
        elements. The element numbering need not be in any specific order. The
        value of node 5 through node 8 has to be zero for Quad4 element to be
        used. Otherwise, Quad8 is automatically chosen.
    emat : array, default=None
        (:math:`n_e*4`) array with element material properties assignment where
        each row is in the form (element number, material number, fiber angle,
        fiber plane angle), where n_e is the total number of elements. The
        element numbering need not be in any specific order. The material
        number corresponds to the materials assigned in the matprops array.

    matprops : array, default=None
        (:math:`nmat*10`) array with the material properties where each row is
        in the form (:math:`E_{11}`, :math:`E_{22}`, :math:`E_{33}`,
        :math:`G_{12}`, :math:`G_{13}`, :math:`G_{23}`, :math:`\nu_{12}`,
        :math:`\nu_{13}`, :math:`\nu_{23}`, :math:`\varrho`), where nmat is the
        total number of different materials considered. The material mechanical
        properties given with respect to the material coordinate system are
        defined as:
            * :math:`E_{11}` the Young modulus of material the 1 direction.
            * :math:`E_{22}` the Young modulus of material the 2 direction.
            * :math:`E_{33}` the Young modulus of material the 3 direction.
            * :math:`G_{12}` the shear modulus in the 12 plane.
            * :math:`G_{13}` the shear modulus in the 13 plane.
            * :math:`G_{23}` the shear modulus in the 23 plane.
            * :math:`\nu_{12}` the Poisson's ratio in the 12 plane.
            * :math:`\nu_{13}` the Poisson's ratio in the 13 plane.
            * :math:`\nu_{23}` the Poisson's ratio in the 23 plane.
            * :math:`\varrho` the material density.
        The rotation of the material constitutive tensor is described in
        Section 3.2.


    load_cases: array
        List of section load vectors to calculate
        stresses, strains and perform failure analysis

    outputs
    -------
    cs_props: array
        cross-sectional properties.
        | standard beam, size: (19)
        | s dm x_cg y_cg ri_x ri_y x_sh y_sh E G I_x I_y K k_x k_y A pitch x_e y_e
        | populated stiffness matrix, size (30)
        | s dm x_cg y_cg ri_x ri_y pitch x_e y_e K_11 K_12 K_13 K_14 K_15 K_16 K_22
        | K_23 K_24 K_25 K_26 K_33 K_34 K_35 K_36 K_44 K_45 K_46 K_55 K_56 K_66
    stress: array
        stresses in each node
    strain: array
        strains in each node
    max_failure: array
        max failure index for each laod case aggregated with simple max function
    max_failure_ks: array
        max failure index for each laod case aggregated with ks function
    """

    def __init__(self, spanpos=0., **kwargs):
        super(BECASWrapper, self).__init__()

        self.basedir = os.getcwd()

        self.dry_run = False
        self.exec_mode = 'octave'
        self.analysis_mode = 'stiffness'
        self.utils_rst_filebase = 'becas_utils'
        self.path_becas = ''
        self.timeout = 180.
        self.path_input = 'becas_inputs/BECAS_SECTION%3.3f' % spanpos
        self.path_plots = 'plots'
        self.checkmesh = False
        self.plot_paraview = True
        self.spanpos = spanpos
        self.hawc2_FPM = False
        self.rho_ks = 50.

        self.nl_2d = np.array([])
        self.el_2d = np.array([], dtype=int)
        self.matprops = np.array([])

        for k, w in kwargs.iteritems():
            try:
                setattr(self, k, w)
            except:
                pass

        # outputs
        if self.hawc2_FPM:
            self.cs_size = 30
            self.cs_props = np.zeros(30)
        else:
            self.cs_size = 19
            self.cs_props = np.zeros(19)
        self.cs_props[0] = spanpos

        self.stress = np.array([])
        self.strain = np.array([])
        self.max_failure = np.array([])
        self.max_failure_ks = np.array([])

    def compute(self):
        """
        execute BECAS using either the Oct2Py bridge or matlab
        """

        tt = time.time()

        try:
            if self.exec_mode == 'oct2py':
                self.execute_oct2py()

            elif self.exec_mode in ['matlab', 'octave']:
                self.execute_shell()
        except:
            if self.hawc2_FPM:
                self.cs_props = np.zeros(30)
            else:
                self.cs_props = np.zeros(19)
            self.cs_props[1] = 1.e6
            # self.max_failure.cases = np.zeros(len(self.load_cases.cases))
        #     self._logger.info('BECAS crashed at R = %2.2f ...' % self.spanpos)

        print ' BECAS calculation time: % 10.6f seconds' % (time.time() - tt)
        # self._logger.info(' BECAS calculation time: % 10.6f seconds' % (time.time() - tt))

    def execute_shell(self):
        """
        Execute BECAS analysis as external program
        """
        out_str = []
        self.setup_path()

        self.utils_rst_filename = os.path.join(self.basedir, self.utils_rst_filebase + '%2.2f.mat' % self.spanpos)
        # self._logger.info('shell execution with analysis_mode = %s' % self.analysis_mode)

        out_str.append('BECAS_SetupPath;\n')
        out_str.append("options.foldername=fullfile('%s');\n" % os.path.join(os.getcwd(), self.path_input))

        if self.analysis_mode in ['stiffness', 'combined']:
            out_str = self.add_utils(out_str)
            out_str = self.add_stiffness_calc(out_str)

        if self.analysis_mode in ['combined', 'stress_recovery']:
            out_str = self.add_stress_recovery(out_str)

        out_str.append('exit;\n')
        self.out_str = out_str

        fid = open('becas_section.m', 'w')
        for i, line in enumerate(out_str):
                fid.write(line)
        fid.close()

        if not self.dry_run:
            if self.exec_mode == 'octave':
                out = commands.getoutput('octave becas_section.m')

            elif self.exec_mode == 'matlab':
                out = commands.getoutput('matlab -nosplash -nodesktop -nojvm -r %s' % 'becas_section')
            print out
            # self._logger.info(out)

            if self.analysis_mode in ['stiffness', 'combined']:
                self.cs_props = np.loadtxt('BECAS2HAWC2.out')
                os.remove('BECAS2HAWC2.out')

        if self.analysis_mode in ['combined', 'stress_recovery']:
            try:
                failure = []
                ks_failure = []
                for i in range(len(self.load_cases.cases)):
                    data = np.loadtxt('failure%i.out' % i)
                    # evaluate KS function of the failure criteria
                    ks_failure.append(ksfunc(data.flatten(), rho=self.rho_ks))
                    # also save the actual max value
                    failure.append(np.max(data))
                self.max_failure.cases = failure
                self.max_failure_ks.cases = ks_failure
            except:
                pass

    def add_utils(self, out_str):

        out_str.append('[ utils ] = BECAS_Utils( options );\n')
        out_str.append('[constitutive.Ks,solutions] = BECAS_Constitutive_Ks(utils);\n')
        if self.plot_paraview:  # and '-fd' not in self.itername:
            path = os.path.join(self.basedir, self.path_plots)
            if not os.path.exists(path):
                os.mkdir(path)
            dirname = os.path.join(path, '%s_span%3.3f' % ('Sec', self.spanpos))
            # self._logger.info('BECAS_PARAVIEW: saving to %s' % dirname)
            out_str.append("BECAS_PARAVIEW('%s', utils);\n" % dirname)
        return out_str

    def add_stiffness_calc(self, out_str):

        out_str.append('[constitutive.Ms] = BECAS_Constitutive_Ms(utils);\n')
        # Check mesh quality
        if self.checkmesh:
            out_str.append('[ meshcheck ] = BECAS_CheckMesh( utils );\n')
        out_str.append('[csprops] = BECAS_CrossSectionProps(constitutive.Ks,utils);\n')
        out_str.append('RadialPosition=%19.12g; \n' % self.spanpos)
        out_str.append("OutputFilename='%s'; \n" % 'BECAS2HAWC2.out')
        out_str.append("utils.hawc2_flag=%s ;\n" % str(not self.hawc2_FPM).lower())
        out_str.append('BECAS_Becas2Hawc2(OutputFilename,RadialPosition,constitutive,csprops,utils)\n')
        out_str.append("save('%s', 'utils', 'solutions', 'csprops')\n" % self.utils_rst_filename)

        return out_str

    def add_stress_recovery(self, out_str):

        # load utils and solutions from saved file
        self.utils_rst_filename = os.path.join(self.basedir, self.utils_rst_filebase + '%2.2f.mat' % self.load_cases.s)
        # self._logger.info('checking for file %s' % self.utils_rst_filename)
        if self.analysis_mode == 'stress_recovery' and os.path.exists(self.utils_rst_filename):
            out_str.append("load('%s', 'utils', 'solutions', 'csprops')\n" % self.utils_rst_filename)
        else:
            raise RuntimeError('utils_rst_filename %s was not found!' % self.utils_rst_filename)

        for i, case in enumerate(self.load_cases.cases):
            load_vector = case._toarray()[[1, 2, 3, 5, 6, 7]]
            if np.sum(load_vector) == 0.:
                np.savetxt('failure%i.out'%i, np.zeros(100))
            else:
                out_str.append('theta0=[%19.12g %19.12g %19.12g %19.12g %19.12g %19.12g]\n' % (load_vector[0],
                                                                                               load_vector[1],
                                                                                               load_vector[2],
                                                                                               load_vector[3],
                                                                                               load_vector[4],
                                                                                               load_vector[5]))
                out_str.append('%Calculate strains\n')
                out_str.append('[strain.GlobalElement,strain.MaterialElement] = BECAS_CalcStrainsElementCenter(theta0,solutions,utils);\n')
                out_str.append('%Calculate stresses\n')
                out_str.append('[ stress.GlobalElement, stress.MaterialElement ] = BECAS_CalcStressesElementCenter( strain, utils );\n')
                out_str.append('%Check failure criteria\n')
                out_str.append('[ failure ] = BECAS_CheckFailure( utils, stress.MaterialElement, strain.MaterialElement );\n')
                out_str.append("FileName='failure%i.out';\n" % i)
                out_str.append("eval(['save ' FileName ' failure -ascii -double']);\n")

            if self.plot_paraview:  # and '-fd' not in self.itername:
                path = os.path.join(self.basedir, self.path_plots)
                if not os.path.exists(path):
                    os.mkdir(path)
                dirname = os.path.join(path, '%s_span%3.3f_case%i' % ('Sec', self.load_cases.s, i))
                # self._logger.info('BECAS_PARAVIEW: saving to %s' % dirname)
                out_str.append("warping=solutions.X*theta0'; \n")
                out_str.append("BECAS_PARAVIEW( '%s', utils, csprops, warping, strain.MaterialElement, stress.MaterialElement, failure )\n"
                    % dirname)
        return out_str

    def setup_path(self):

        setup_path=("function BECAS_SetupPath\n"
                    "addpath(genpath(fullfile('%s','BECAS_elemlib')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_examples')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_fea')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_solve')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_post')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_main')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_other')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_pre')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_strength')))\n"
                    "addpath(genpath(fullfile('%s','BECAS_crack')))"
                    "end\n"%(self.path_becas, self.path_becas, self.path_becas, self.path_becas,
                             self.path_becas, self.path_becas, self.path_becas, self.path_becas,
                             self.path_becas, self.path_becas))

        fid = open('BECAS_SetupPath.m','w')
        fid.write(setup_path)
        fid.close()

    def execute_oct2py(self):
        """
        Execute BECAS analysis
        ======================

        Uses oct2py Octave to Python bridge to execute all required analysis
        in BECAS.

        This way of executing BECAS was abandoned since it had some weird
        conflict with OpenMDAO's CaseIteratorDriver - probably due to the fact
        that both CaseIteratorDriver and Oct2Py use multiprocessing.
        """

        from oct2py import Oct2Py

        self.load_input_vars()

        def isNoneType(x):
            if x is None:
                return True
            else:
                return False

        # check if the path to the BECAS program is properly defined
        if self.path_becas is '':
            msg = "path_becas is empty, please define a valid absolute path to BECAS"
            raise ValueError, msg

        # self._logger.info('executing BECAS ...')

        # to run concurrently you need a unique instance of oct2py, ie Oct2Py()
        # but this still seems to leave old instances of octave floating around
        # self.octave = octave
        t0 = time.time()
        self.octave = Oct2Py()
        # self.octave = Oct2Py(logger=self._logger)
        # self._logger.info('getting Oct2Py instance: % 10.6f seconds' % (time.time() - t0))
        # this will produce a lot of output to the openmdao_log.txt file
        # self.octave = Oct2Py(logger=self._logger)
        self.octave.timeout = self.timeout

        # short hand notation
        oc = self.octave.run
        # set working dir of Octave to the BECAS source folder
        # oc("cd('%s')" % self.path_becas)

        t0 = time.time()
        self.setup_path()
        oc('BECAS_SetupPath')
        # self._logger.info('BECAS_SetupPath: % 10.6f seconds' % (time.time() - t0))

        try:
            t0 = time.time()
            # Build arrays for BECAS
            # use BECAS input file loader when there is valid input path defined
            if self.path_input is not '':
                oc("options.foldername='%s'" % os.path.join(os.getcwd(), self.path_input))
                oc("[utils] = BECAS_Utils(options);")
            else:
                # make sure we have the inputs defined correctly
                seq = (self.nl_2d,self.el_2d,self.emat,self.matprops)
                if any(map(isNoneType, seq)):
                    raise ValueError, 'Not all BECAS inputs are defined'
                self.put_input_vars()
                oc("[utils] = BECAS_Utils(options, nl_2d, el_2d, emat, matprops);")
            # self._logger.info('BECAS_Utils: % 10.6f seconds' % (time.time() - t0))
            # Check mesh quality
            if self.checkmesh:
                oc("[ meshcheck ] = BECAS_CheckMesh( utils );")

            # BECAS module for the evaluation of the cross section stiffness matrix
            t0 = time.time()
            oc("[ constitutive.Ks, solutions ] = BECAS_Constitutive_Ks(utils);")
            # self._logger.info('BECAS_Constitutive_Ks: % 10.6f seconds' % (time.time() - t0))

            t0 = time.time()
            # BECAS module for the evaluation of the cross section mass matrix
            # self._logger.info('BECAS_Constitutive_Ms: % 10.6f seconds' % (time.time() - t0))
            oc("[ constitutive.Ms ] = BECAS_Constitutive_Ms(utils);")
            t0 = time.time()
            # BECAS module for the evaluation of the cross section properties
            # self._logger.info('BECAS_CrossSectionProps: % 10.6f seconds' % (time.time() - t0))
            oc("[ csprops ] = BECAS_CrossSectionProps(constitutive.Ks, utils);")

            # Output of results to HAWC2 st file
            t0 = time.time()
            oc("RadPos=1;") # Define radial position
            inputs = 'false, RadPos, constitutive, csprops, utils,' + str(self.hawc2_FPM).lower()
            oc("[cs_props] = BECAS_Becas2Hawc2(%s);" % inputs)
            # self._logger.info('BECAS_Becas2Hawc2: % 10.6f seconds' % (time.time() - t0))
            self.paraview_octave()
            # obtain the output variables from Octave
            t0 = time.time()
            self.get_output_vars()
            # self._logger.info('get_output_vars: % 10.6f seconds' % (time.time() - t0))
            # compute stresses and strains and check for failures
            t0 = time.time()
            self.stress_recovery_octave()
            # self._logger.info('stress_recovery: % 10.6f seconds' % (time.time() - t0))
        except:
            if self.hawc2_FPM:
                h2c = np.zeros(30)
            else:
                h2c = np.zeros(19)
            h2c[1] = 2.e3
            self.paraview_octave(force=True)
            # self._logger.info('BECAS crashed ...')
        self.octave.close()

    def load_input_vars(self):
        """
        Load BECAS input files from a directory. This is entirely optional.
        """

        # optionally load a set of input files
        self.nl_2d = np.loadtxt(os.path.join(self.path_input, 'N2D.in') )
        self.el_2d = np.loadtxt(os.path.join(self.path_input, 'E2D.in') )
        self.emat = np.loadtxt(os.path.join(self.path_input, 'EMAT.in') )
        self.matprops = np.loadtxt(os.path.join(self.path_input,'MATPROPS.in'))


    def get_output_vars_oct2py(self):
        """
        Obtain all BECAS output variables through Octave
        """

        # These are nested variables in Octave, they become dictionaries
        # in Python.

        self.utils        = self.octave.get('utils')
        self.csprops      = self.octave.get('csprops')
        self.constitutive = self.octave.get('constitutive')
        if self. checkmesh:
            self.meshcheck    = octave.get('meshcheck')
#        self.options      = octave.get('options')
#        self.solutions    = octave.get('solutions')
#        self.strain       = octave.get('strain')
#        self.stress       = octave.get('stress')
        self.cs_props  = self.octave.get('cs_props')
        h2c = self.cs_props[0]

    def put_input_vars_oct2py(self):
        """
        Put all input variables for BECAS into Octave
        """

        self.octave.put('nl_2d', self.nl_2d)
        self.octave.put('el_2d', self.el_2d)
        self.octave.put('emat', self.emat)
        self.octave.put('matprops', self.matprops)

    def mesh_check_oct2py(self):
        oc = self.octave.run
        oc("[ meshcheck ] = BECAS_CheckMesh( utils );")
        self.meshcheck    = self.octave.get('meshcheck')

    def stress_recovery_oct2py(self):
        """

        Parameters
        ----------

        loadvector : ndarray(6)
            Forces and moments in x, y and z directions

        """

        if len(self.load_cases.cases) == 0:
            return

        # short hand notation
        oc = self.octave.run

        # Recover strains and stress for a list of force and moment vectors
        # and extract failure criteria for each case.
        self.failure_elements = []
        failure = []
        ks_failure = []
        tstrain = 0.
        tstress = 0.
        tfailure = 0.
        for i, case in enumerate(self.load_cases.cases):
            load_vector = case._toarray()[1:]
            self.octave.put('theta0', load_vector)
            # Calculate strains
            t1 = time.time()
            oc("[ strain ] = BECAS_RecoverStrains(theta0, solutions, utils)")
            tstrain += time.time() - t1
            # self.strain = self.octave.get('strain')
            # Calculate stresses
            t1 = time.time()
            oc("[ stress ] = BECAS_RecoverStresses( strain, utils )")
            tstress += time.time() - t1

            # self.stress = self.octave.get('stress')
            # ipdb.set_trace()
            t1 = time.time()
            oc("[ failure ] = BECAS_CheckFailure( utils, stress.MaterialElement, strain.MaterialElement )")
            tfailure += time.time() - t1

            f = self.octave.get('failure')
            self.failure_elements.append(f)
            failure.append(np.max(f))
            ks_failure.append(ksfunc(f.flatten(), rho=self.rho_ks))

            if self.plot_paraview:
                if 'fd' in self.parent.itername:
                    continue
                path = os.path.join(self.basedir, self.path_plots)
                if not os.path.exists(path):
                    os.mkdir(path)
                dirname = os.path.join(path, 'BECAS_PARAVIEW%s_case%02d' % (self.itername, i))
                # self._logger.info('BECAS_PARAVIEW: saving to %s' % dirname)
                solutions = self.octave.get('solutions')
                self.octave.put('warping', solutions['X'] * load_vector)
                oc("BECAS_PARAVIEW('%s', utils, csprops, warping, strain.MaterialElement, stress.MaterialElement, failure )" % dirname)
        self.max_failure.cases = failure
        self.max_failure_ks.cases = ks_failure
        # self._logger.info('BECAS_RecoverStrains: %10.6f seconds' % tstrain)
        # self._logger.info('BECAS_RecoverStresses: %10.6f seconds' % tstress)
        # self._logger.info('BECAS_CheckFailure: %10.6f seconds' % tfailure)

    def paraview_oct2py(self, force=False):

        if self.plot_paraview:
            if 'fd' in self.parent.itername and not force:
                return
            path = os.path.join(self.basedir, self.path_plots)
            if not os.path.exists(path):
                os.mkdir(path)
            dirname = os.path.join(path, 'BECAS_PARAVIEW%s' % self.itername)
            # self._logger.info('BECAS_PARAVIEW: saving to %s' % dirname)
            self.octave.run("BECAS_PARAVIEW('%s', utils)" % dirname)


    def plot_mesh(self, ax, element_nr=True):
        """
        based on BECAS_post/BECAS_PlotElements.m
        """
        if (self.utils.etype == 1):
            vertex_connection = np.array([1, 2, 3, 4]) - 1
        elif (self.utils.etype == 2 or self.utils.etype == 3):
            vertex_connection = np.array([1, 5, 2, 6, 3, 7, 4, 8]) - 1
        elif (self.utils.etype == 4):
            vertex_connection = np.array([1, 2, 3]) - 1

        # at this point we have loaded results from Matlab and hence the node
        # and element numbering is all one based!
        ut = self.utils
        vertex = np.ndarray(shape=(ut.nnpe_2d, 2))
        for i in xrange((int(ut.ne_2d))):
            for ii in xrange((int(ut.nnpe_2d))):
                # ut.mapel_2d : el_nr, nr_1, nr_2, nr_3, ..., nr_nnpe_2d
                vertex[ii, :] = ut.nl_2d[ut.mapel_2d[i, ii+1]-1, 1:3]
            # prepare the element lines as a polygone patch
            poly = mpl.patches.Polygon(vertex[vertex_connection,:],
                                       fc='none', ec='r', ls='dashed')
            # draw the patch on the axis
            ax.add_patch(poly)

            # place the text somewhere in the element
            if element_nr:
                x, y = vertex[:,0].mean(), vertex[:,1].mean()
                ax.annotate('%i' % (i+1), xy=(x,y), xycoords='data', fontsize=8,
                            xytext=(+0, +0), textcoords='offset points')

        return ax

    def _plot_mesh_multi_elements(self):
        """

        This version will support future BECAS versions that have support for
        multiple element types in a single mesh.

        based on BECAS_post/BECAS_PlotElements.m
        """

        # TODO: this function is not ready yet, and the octave BECAS branch
        # will not work with this function

        # BECAS has now support for multiple elements, so each element has a
        # type flag. For now, only support meshes with one element type
        if np.all(self.utils.etype.__eq__(1)):
            vertex_connection = np.array([1, 2, 3, 4])
        elif np.all(self.utils.etype.__eq__(2)) or np.all(self.utils.etype.__eq__(3)):
            vertex_connection = np.array([1, 5, 2, 6, 3, 7, 4, 8])
        elif np.all(self.utils.etype.__eq__(4)):
            vertex_connection = np.array([1, 2, 3])
        else:
            raise ValueError, 'Multiple element types not supported yet.'
        print vertex_connection

#%Start looping elements
#iv=0;
#for i=1:utils.ne_2d
#    %Defining constants to make equations readable
#    nnpe_2d = utils.element(utils.etype(i)).nnpe_2d;
#    vertex_connection = utils.element(utils.etype(i)).vertex_connection;
#
#    vertex_list=zeros(nnpe_2d,3);
#    for ii=1:nnpe_2d
#        iv=iv+1;
#        for iii=1:2
#            vertex_list(ii,iii)=utils.nl_2d(utils.mapel_2d(i,ii+1),iii+1);
#        end
#        vertex_list(ii,3)=0;
#        %Plot node numbers
#%         text(utils.nl_2d(utils.mapel_2d(i,ii+1),2),
#        utils.nl_2d(utils.mapel_2d(i,ii+1),3),1,num2str(utils.mapel_2d(i,ii+1)));
#    end
#    patch('Vertices',vertex_list,'Faces',vertex_connection,...
#        'FaceColor',[1 1 1],'EdgeColor',[0. 0. 0.],'FaceAlpha', 0);
#end

        ut = self.utils
        # Start looping elements
        iv=0
        patches = []
        for i in xrange(ut.ne_2d):
            # Defining constants to make equations readable
            iel = ut.utils.etype[i] -1 # -1: convert to zero based index
            nnpe_2d = ut.element.nnpe_2d[iel]
            vertex_connection = ut.element.vertex_connection[iel]
            vertex_list = np.zeros( (nnpe_2d,3) )
            for ii in xrange(nnpe_2d):
                iv=iv+1
                for iii in [0,1]:#1:2
                    vertex_list[ii,iii] = ut.nl_2d[ut.mapel_2d[i,ii+1],iii+1]
                vertex_list[ii,3-1] = 0
                # Plot node numbers
                #text(utils.nl_2d(utils.mapel_2d(i,ii+1),2),
                #utils.nl_2d(utils.mapel_2d(i,ii+1),3),1,num2str(utils.mapel_2d(i,ii+1)));

                #Plot node numbers
                #text(utils.nl_2d(utils.mapel_2d(i,ii+1),2),
                #utils.nl_2d(utils.mapel_2d(i,ii+1),3),1,
                #num2str(utils.mapel_2d(i,ii+1)));

            patches.append([vertex_list, vertex_connection])

        return patches


if __name__ == '__main__':
    pass
