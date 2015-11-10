
import time
import numpy as np
import unittest

from openmdao.core.mpi_wrap import MPI
from openmdao.api import Problem, Group, IndepVarComp

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure
from becas_wrapper.becas_bladestructure import BECASBeamStructure



# stuff for running in parallel under MPI
def mpi_print(prob, *args):
    """ helper function to only print on rank 0"""
    if prob.root.comm.rank == 0:
        print(args)

if MPI:
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.core.basic_impl import BasicImpl as impl

from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       SplinedBladePlanform, \
                                       PGLRedistributedPlanform


def configure(nsec_st, dry_run=False, FPM=False, with_sr=False):

    p = Problem(impl=impl, root=Group())

    p.root.add('blade_length_c', IndepVarComp('blade_length', 86.366), promotes=['*'])

    pfo = read_blade_planform('data/DTU_10MW_RWT_blade_axis_prebend.dat')
    nsec_ae = 50
    s_ae = np.linspace(0, 1, nsec_ae)
    s_st = np.linspace(0, 1, nsec_st)
    pf = {}
    pf['s'] = s_ae
    for k, v in pfo.iteritems():
        if k in ['s', 'smax', 'blade_length']: continue
        pf[k] = np.interp(s_ae, pfo['s'], v)

    spl = p.root.add('pf_splines', SplinedBladePlanform(pf), promotes=['*'])
    spl.configure()
    redist = p.root.add('pf_st', PGLRedistributedPlanform('_st', nsec_ae, s_st), promotes=['*'])

    cfg = {}
    cfg['redistribute_flag'] = False
    cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])
    afs = []
    for f in ['data/ffaw3241.dat',
              'data/ffaw3301.dat',
              'data/ffaw3360.dat',
              'data/cylinder.dat']:

        afs.append(np.loadtxt(f))
    cfg['base_airfoils'] = afs
    surf = p.root.add('blade_surf', PGLLoftedBladeSurface(cfg, size_in=nsec_st,
                                    size_out=(200, nsec_st, 3), suffix='_st'), promotes=['*'])
    surf._dry_run = True
    # read the blade structure
    st3d = read_bladestructure('data/DTU10MW')

    # and interpolate onto new distribution
    st3dn = interpolate_bladestructure(st3d, s_st)

    spl = p.root.add('st_splines', SplinedBladeStructure(st3dn), promotes=['*'])
    spl.add_spline('DP04', np.linspace(0, 1, 4), spline_type='pchip')
    spl.add_spline('DP05', np.linspace(0, 1, 4), spline_type='pchip')
    spl.add_spline('DP08', np.linspace(0, 1, 4), spline_type='pchip')
    spl.add_spline('DP09', np.linspace(0, 1, 4), spline_type='pchip')
    spl.add_spline('r04uniaxT', np.linspace(0, 1, 4), spline_type='pchip')
    spl.add_spline('r08uniaxT', np.linspace(0, 1, 4), spline_type='pchip')
    spl.configure()
    # inputs to CS2DtoBECAS and BECASWrapper
    config = {}
    cfg = {}
    cfg['dry_run'] = dry_run
    cfg['path_shellexpander'] = '/Users/frza/git/BECAS_stable/shellexpander/shellexpander'
    cfg['dominant_elsets'] = ['REGION04', 'REGION08']
    cfg['max_layers'] = 0
    config['CS2DtoBECAS'] = cfg
    cfg = {}
    cfg['path_becas'] = '/Users/frza/git/BECAS_stable/BECAS/src/matlab'
    cfg['hawc2_FPM'] = FPM
    cfg['dry_run'] = dry_run
    cfg['analysis_mode'] = 'stiffness'
    config['BECASWrapper'] = cfg

    p.root.add('stiffness', BECASBeamStructure(p.root, config, st3dn, (200, nsec_st, 3)), promotes=['*'])
    p.setup()
    for k, v in pf.iteritems():
        if k in p.root.pf_splines.params.keys():
            p.root.pf_splines.params[k] = v

    # p['hub_radius'] = 2.8
    # p['blade_x'] = d.pf['x'] * 86.366
    # p['blade_z'] = d.pf['y'] * 86.366
    # p['blade_y'] = d.pf['z'] * 86.366
    return p

class BECASWrapperTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # add some tests here...

    def test_dry_run(self):

        p = configure(4, True)
        p.run()

    def test_stiffness_run(self):

        p = configure(4, False, False)
        p.run()
        self.assertEqual(np.testing.assert_array_almost_equal(p['beam_structure'], beam_st, decimal=4), None)

        self.assertAlmostEqual(p['blade_mass'], 42501.309314525213, places=6)
        self.assertAlmostEqual(p['blade_mass_moment'], 1024100.0975330817, places=6)

    # def test_FPM(self):
    #
    #     p = configure(4, False, True)
    #     p.setup()
    #     p.run()
    #     self.assertEqual(np.testing.assert_array_almost_equal(p['beam_structure'], beam_st_FPM, decimal=6), None)

    # def test_stiffness_and_stress_recovery_run(self):
    #
    #     p = configure(4, False, False, True)
    #     p.setup()
    #     p.run()
    #     self.assertEqual(np.testing.assert_array_almost_equal(p['beam_structure'], beam_st, decimal=4), None)


if __name__ == "__main__":

    # unittest.main()
    t0 = time.time()
    # numer of components in ParallelGroup
    nsec_st = 8
    p = configure(nsec_st, True, False)
    t1 = time.time()
    p.run()
    t2 = time.time()
    print 'setup: ', t1 - t0
    print 'run: ', t2 - t1
    # print('mass %f'%p['blade_mass'])
