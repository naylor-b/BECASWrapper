
import time
import numpy as np
import unittest

from openmdao.core.mpi_wrap import MPI
from openmdao.core import Problem, Group

from fusedwind.turbine.structure import read_bladestructure, interpolate_bladestructure
from becas_wrapper.becas_bladestructure import BECASBeamStructure

beam_st_FPM = np.array([[  0.0000000000000000e+00,   1.1952118833699999e+03,
         -1.5858113855000001e-04,  -3.0099069983299998e-04,
          1.7613104630900001e+00,   1.8320413152299999e+00,
         -7.5141552360899993e+01,  -9.1237708162299994e-05,
         -3.3068965130699998e-04,   2.5620766117500000e+09,
         -3.2713188485800000e+06,   0.0000000000000000e+00,
          0.0000000000000000e+00,   0.0000000000000000e+00,
         -1.1699087846700000e+07,   1.9250772466400001e+09,
          0.0000000000000000e+00,   0.0000000000000000e+00,
          0.0000000000000000e+00,   1.1705779708799999e+06,
          1.7853152912400002e+10,   1.5289645902200001e-03,
         -7.9791192946100003e-04,   0.0000000000000000e+00,
          6.1257694173400002e+10,  -1.1825561523400000e-03,
          0.0000000000000000e+00,   6.2358618243599998e+10,
          0.0000000000000000e+00,   2.7717382853799999e+10],
       [  3.3333333333300003e-01,   5.9240523824900004e+02,
         -2.6693986375800000e-01,   2.8489921897899999e-02,
          7.5224534722199998e-01,   1.5513120103200000e+00,
         -4.1011479101299999e+00,  -1.1434517365000001e-01,
          2.2906248698800001e-02,   5.9101867760300004e+08,
         -4.2581746486599997e+07,   0.0000000000000000e+00,
          0.0000000000000000e+00,   0.0000000000000000e+00,
         -7.8805439582399994e+07,   4.8143702880400002e+08,
          0.0000000000000000e+00,   0.0000000000000000e+00,
          0.0000000000000000e+00,   2.9313611627200001e+08,
          8.8377915961900005e+09,  -3.7517132169400001e-03,
          1.0860630726900000e-03,   0.0000000000000000e+00,
          5.9235918684399996e+09,  -1.7213821411100000e-03,
          0.0000000000000000e+00,   1.9242877307500000e+10,
          0.0000000000000000e+00,   1.3627878469200001e+09],
       [  6.6666666666700003e-01,   2.7979129555899999e+02,
         -1.3085432079100001e-01,   2.7899029556200002e-02,
          3.5473343513500000e-01,   8.9231989775300002e-01,
         -1.1055091524400000e+00,  -2.9282263478599999e-02,
          2.4369157458600001e-02,   2.9434182411799997e+08,
         -2.4517411236800002e+06,   0.0000000000000000e+00,
          0.0000000000000000e+00,   0.0000000000000000e+00,
         -1.1486259543099999e+07,   1.8892705726699999e+08,
          0.0000000000000000e+00,   0.0000000000000000e+00,
          0.0000000000000000e+00,   5.7832383329599999e+07,
          4.5774631376599998e+09,  -3.7573326641400002e-04,
         -9.0101565309200001e-05,   0.0000000000000000e+00,
          6.5852507620099998e+08,   8.7523832917199996e-05,
          0.0000000000000000e+00,   3.1051489975200000e+09,
          0.0000000000000000e+00,   1.4959545997000000e+08],
       [  1.0000000000000000e+00,   9.7004310002600000e+00,
         -7.8911856188899995e-02,   6.1158418429599999e-03,
          4.1991736168599997e-02,   1.7111989411600001e-01,
         -8.8095762348399997e-01,  -6.9612164890700004e-02,
          5.6772016249600002e-03,   2.4162179034299999e+07,
          3.4251886333399999e+05,   0.0000000000000000e+00,
          0.0000000000000000e+00,   0.0000000000000000e+00,
         -4.4413826677400000e+04,   6.4912673721599998e+06,
          0.0000000000000000e+00,   0.0000000000000000e+00,
          0.0000000000000000e+00,   6.5951164472099999e+05,
          1.1305527702500001e+08,  -1.1743630908100000e-07,
         -4.1908763445600000e-07,   0.0000000000000000e+00,
          2.3011353774999999e+05,   3.6865912989000003e-08,
          0.0000000000000000e+00,   3.3669232130000000e+06,
          0.0000000000000000e+00,   2.3166614213500000e+05]])

beam_st = np.array([[  0.0000000000000000e+00,   1.1952118833699999e+03,
         -1.5858113855000001e-04,  -3.0099069983299998e-04,
          1.7613104630900001e+00,   1.8320413152299999e+00,
          4.4755272199400002e-03,   2.5980958243700000e-04,
          1.2615910178600000e+10,   2.5349657195700002e+09,
          4.8555905444800000e+00,   4.9428552803999999e+00,
          1.0934005349700000e+01,   5.4786041573400002e-01,
          7.0298181091000000e-01,   1.4151299953500001e+00,
         -7.5141552360899993e+01,  -9.1237708162299994e-05,
         -3.3068965130699998e-04],
       [  3.3333333333300003e-01,   5.9240523824900004e+02,
         -2.6693986375800000e-01,   2.8489921897899999e-02,
          7.5224534722199998e-01,   1.5513120103200000e+00,
          4.9146905983400002e-01,   6.9743597891799999e-02,
          8.0845171311199999e+09,   1.4781728669300001e+09,
          7.3270818434399998e-01,   2.3802135607400001e+00,
          7.9797333509900004e-01,   3.6164524923899999e-01,
          3.0204351832400000e-01,   1.0931749482199999e+00,
         -4.1011479101299999e+00,  -1.1434517365000001e-01,
          2.2906248698800001e-02],
       [  6.6666666666700003e-01,   2.7979129555899999e+02,
         -1.3085432079100001e-01,   2.7899029556200002e-02,
          3.5473343513500000e-01,   8.9231989775300002e-01,
          2.7700083977300000e-01,   5.4943271833600003e-02,
          1.2241669888600000e+10,   2.1525101425900002e+09,
          5.3793729302599999e-02,   2.5365403787000002e-01,
          6.1091836204499997e-02,   3.6553172339500001e-01,
          2.3489419758700000e-01,   3.7392473243399998e-01,
         -1.1055091524400000e+00,  -2.9282263478599999e-02,
          2.4369157458600001e-02],
       [  1.0000000000000000e+00,   9.7004310002600000e+00,
         -7.8911856188899995e-02,   6.1158418429599999e-03,
          4.1991736168599997e-02,   1.7111989411600001e-01,
          3.2199194863100002e-02,   7.3929247375999997e-03,
          7.0793549315600004e+09,   2.1852900934600000e+09,
          3.2504873674900003e-05,   4.7559745846199999e-04,
          7.5230278810899996e-05,   6.9253869055100004e-01,
          1.8582234314500001e-01,   1.5969714489299999e-02,
         -8.8095762348399997e-01,  -6.9612164890700004e-02,
          5.6772016249600002e-03]])



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
    from openmdao.core import BasicImpl as impl

from PGL.components.loftedblade import LoftedBladeSurface
from PGL.main.planform import read_blade_planform, redistribute_planform


def configure(nsec, dry_run=False, FPM=False, with_sr=False):
    pf = read_blade_planform('data/DTU_10MW_RWT_blade_axis_prebend.dat')

    s_new = np.linspace(0, 1, nsec)

    pf = redistribute_planform(pf, s=s_new)

    d = LoftedBladeSurface()
    d.pf = pf
    d.redistribute_flag = False
    # d.minTE = 0.0002

    d.blend_var = [0.241, 0.301, 0.36, 1.0]
    for f in ['data/ffaw3241.dat',
              'data/ffaw3301.dat',
              'data/ffaw3360.dat',
              'data/cylinder.dat']:

        d.base_airfoils.append(np.loadtxt(f))

    d.update()
    d.surface *= 86.366
    d.surfnorot *= 86.366

    # read the blade structure
    st3d = read_bladestructure('data/DTU10MW')

    # and interpolate onto new distribution
    st3dn = interpolate_bladestructure(st3d, s_new)

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

    p = Problem(impl=impl, root=Group())
    p.root.add('stiffness', BECASBeamStructure(config, st3dn, d.surfnorot), promotes=['*'])
    p.setup()
    p['hub_radius'] = 2.8
    p['blade_x'] = d.pf['x'] * 86.366
    p['blade_z'] = d.pf['y'] * 86.366
    p['blade_y'] = d.pf['z'] * 86.366
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

    unittest.main()
    # p = configure(4, False, True)
    # p.run()
