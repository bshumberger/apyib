import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *


def test_mp2_aat_fc():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    I_00_ref = np.array(
    [[-0.000000000000885, -0.000000000000133, -0.221388055285698],
     [-0.000000000001037,  0.000000000001387,  0.000000000000555],
     [ 0.322447592891834, -0.000000000008758,  0.000000000000014],
     [ 0.000000000000298, -0.00000000000149 ,  0.058593758263478],
     [ 0.000000000003303, -0.00000000000104 , -0.133537080585562],
     [-0.224221200569403,  0.211180967869124, -0.000000000000488],
     [ 0.000000000000989,  0.000000000001123,  0.058593758265735],
     [-0.000000000001327, -0.000000000000625,  0.133537080582084],
     [-0.224221200546015, -0.211180967865547,  0.000000000000476]])

    I_0D_ref = np.array(
    [[ 0.000000000000018,  0.000000000000002,  0.006933445607431],
     [ 0.000000000000046, -0.000000000000018,  0.000000000000036],
     [-0.006425064384323,  0.000000000000112, -0.               ],
     [ 0.000000000000013,  0.000000000000019, -0.001737179355382],
     [-0.000000000000047,  0.000000000000013,  0.004157552320948],
     [ 0.004467813160596, -0.002705804054262, -0.000000000000001],
     [-0.000000000000038, -0.000000000000014, -0.001737179355417],
     [ 0.000000000000045,  0.000000000000008, -0.004157552320787],
     [ 0.004467813160123,  0.0027058040542  ,  0.000000000000001]])

    I_D0_ref = np.array(
    [[-0.000000000000023, -0.000000000000066, -0.006933445527731],
     [-0.000000000000015,  0.000000000000016,  0.000000000000018],
     [ 0.006425064333186, -0.000000000000106,  0.               ],
     [ 0.000000000000041,  0.000000000000015,  0.001737179409087],
     [ 0.000000000000071,  0.00000000000004 , -0.004157552391643],
     [-0.004467813570512,  0.002705804364734, -0.000000000000007],
     [-0.000000000000024,  0.000000000000067,  0.001737179409158],
     [-0.000000000000026, -0.000000000000048,  0.00415755239134 ],
     [-0.004467813569996, -0.002705804364666,  0.000000000000007]])

    I_DD_ref = np.array(
    [[ 0.000000000000033, -0.00000000000002 , -0.003120426021844],
     [-0.000000000000028,  0.000000000000035, -0.000000000000007],
     [ 0.013918781838518, -0.000000000000272,  0.               ],
     [-0.000000000000013, -0.000000000000053,  0.000463223383833],
     [ 0.000000000000033, -0.000000000000008, -0.003126854667737],
     [-0.008189297494473,  0.006450493269525,  0.000000000000003],
     [-0.000000000000056,  0.000000000000071,  0.000463223383812],
     [-0.000000000000012, -0.00000000000002 ,  0.003126854667669],
     [-0.008189297493953, -0.006450493269404, -0.000000000000003]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

def test_cid_aat_fc():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    I_00_ref = np.array(
    [[-0.000000000000865, -0.00000000000013 , -0.216366014494262],
     [-0.000000000001013,  0.000000000001356,  0.000000000000542],
     [ 0.315133083699127, -0.000000000008559,  0.000000000000013],
     [ 0.000000000000291, -0.000000000001456,  0.057264597838427],
     [ 0.000000000003228, -0.000000000001017, -0.130507880750658],
     [-0.219134891764742,  0.206390468082444, -0.000000000000477],
     [ 0.000000000000967,  0.000000000001097,  0.057264597840633],
     [-0.000000000001297, -0.000000000000611,  0.130507880747258],
     [-0.219134891741894, -0.206390468078957,  0.000000000000465]])

    I_0D_ref = np.array(
    [[ 0.000000000000023,  0.000000000000002,  0.008711373870197],
     [ 0.00000000000006 , -0.000000000000017,  0.000000000000125],
     [-0.008224150533593,  0.000000000000105, -0.               ],
     [ 0.000000000000016,  0.000000000000018,  0.001145949219258],
     [-0.000000000000062,  0.000000000000012,  0.004016041319127],
     [ 0.005718848172509, -0.002534518326999, -0.000000000000001],
     [-0.000000000000049, -0.000000000000013,  0.001145949219013],
     [ 0.000000000000056,  0.000000000000008, -0.004016041319151],
     [ 0.005718848171899,  0.002534518326931,  0.000000000000001]])

    I_D0_ref = np.array(
    [[ 0.000000000000014, -0.000000000000037, -0.00871137362683 ],
     [-0.000000000000014,  0.000000000000018, -0.000000000000063],
     [ 0.008224150538242, -0.000000000000099,  0.               ],
     [ 0.000000000000019, -0.000000000000005, -0.001145949266879],
     [ 0.000000000000056,  0.000000000000017, -0.004016041310459],
     [-0.005718848825692,  0.002534518850514, -0.000000000000007],
     [-0.00000000000006 ,  0.000000000000045, -0.001145949266655],
     [-0.               , -0.000000000000025,  0.004016041310337],
     [-0.005718848825031, -0.002534518850439,  0.000000000000007]])

    I_DD_ref = np.array(
    [[ 0.000000000000067, -0.000000000000044, -0.006326297657552],
     [-0.000000000000004,  0.000000000000126,  0.000000000000002],
     [ 0.034475163791576, -0.000000000000737,  0.               ],
     [-0.000000000000097, -0.000000000000121,  0.001213393111575],
     [ 0.000000000000051, -0.000000000000021, -0.009560380543719],
     [-0.019735452963484,  0.01602643832959 ,  0.000000000000002],
     [-0.000000000000044,  0.000000000000161,  0.00121339311152 ],
     [-0.000000000000067, -0.000000000000036,  0.009560380543523],
     [-0.019735452961626, -0.016026438329409, -0.000000000000002]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)



