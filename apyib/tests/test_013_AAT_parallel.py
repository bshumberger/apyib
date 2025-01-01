import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_parallel_rhf_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference AAT.
    aat_ref = np.array(
    [[ 0.00000000000000,    0.00000000000000,   -0.22630653868957],
     [-0.00000000000000,   -0.00000000000000,    0.00000000000000],
     [ 0.32961125700525,   -0.00000000000000,   -0.00000000000000],
     [-0.00000000000000,   -0.00000000000000,    0.05989549730400],
     [ 0.00000000000000,    0.00000000000000,   -0.13650378268362],
     [-0.22920257093325,    0.21587263338256,   -0.00000000000000],
     [-0.00000000000000,   -0.00000000000000,    0.05989549730400],
     [-0.00000000000000,   -0.00000000000000,    0.13650378268362],
     [-0.22920257093325,   -0.21587263338256,    0.00000000000000]])

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

def test_mp2_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference AAT.
    I_00_ref = np.array(
    [[-0.097856900360615, -0.024464664951138,  0.069232106541966],
     [ 0.024686227422983,  0.005879922586945, -0.003819266089423],
     [-0.209265694313652, -0.051803561259253,  0.093935959595289],
     [-0.088710310616063, -0.022263494739757,  0.060278744166398],
     [-0.016456264690831, -0.004056816513517,  0.020292183723423],
     [-0.215140025600313, -0.05317143139671 ,  0.089029423759874],
     [-0.088710310631667, -0.022263494743767, -0.060278744172876],
     [-0.016456264697269, -0.004056816515104, -0.020292184243793],
     [ 0.215140025582485,  0.053171431392279,  0.089029423753599],
     [-0.097856900385283, -0.02446466495731 , -0.069232106543105],
     [ 0.024686227444831,  0.005879922592439,  0.00381926608413 ],
     [ 0.209265694510914,  0.051803561307904,  0.093935959578609]])
    
    I_0D_ref = np.array(
    [[ 0.006808740805464,  0.001695403965524, -0.005130119220599],
     [-0.000871959182927, -0.00021540761311 , -0.000115195855897],
     [ 0.013071058381498,  0.003251728200012, -0.008080999183209],
     [ 0.006420223678521,  0.001599163975967, -0.004808528656272],
     [ 0.00097884078409 ,  0.00024339858343 , -0.00054627260451 ],
     [ 0.013187776617029,  0.003280199759361, -0.007966306041871],
     [ 0.006420223679836,  0.001599163976289,  0.004808528656924],
     [ 0.000978840784731,  0.0002433985836  ,  0.000546272620396],
     [-0.013187776615873, -0.003280199759082, -0.007966306041163],
     [ 0.006808740806848,  0.001695403965874,  0.005130119220869],
     [-0.000871959184344, -0.000215407613452,  0.000115195856348],
     [-0.013071058393264, -0.003251728202942, -0.008080999183206]])
    
    I_D0_ref = np.array(
    [[-0.006808740892117, -0.001695403973672,  0.005130119233505],
     [ 0.0008719591981  ,  0.000215407614326,  0.000115195854473],
     [-0.013071058531117, -0.003251728212792,  0.008080999226863],
     [-0.006420223760504, -0.001599163986204,  0.004808528668143],
     [-0.000978840803084, -0.000243398587652,  0.000546272606263],
     [-0.01318777676535 , -0.003280199772866,  0.007966306084481],
     [-0.006420223761249, -0.001599163986388, -0.004808528668831],
     [-0.000978840803297, -0.000243398587705, -0.000546272622777],
     [ 0.013187776763665,  0.003280199772445,  0.007966306084511],
     [-0.006808740893785, -0.001695403974086, -0.005130119232304],
     [ 0.000871959199832,  0.000215407614753, -0.000115195854557],
     [ 0.013071058542963,  0.003251728215738,  0.008080999227038]])
    
    I_DD_ref = np.array(
    [[ 0.000030731202513,  0.000025700510925, -0.000056151104364],
     [-0.000014926915508, -0.000001173748787, -0.000005567152441],
     [ 0.000200327181323,  0.000027280454078, -0.000071478160397],
     [ 0.000006267830815, -0.000012114425716,  0.000026462669104],
     [ 0.000055984628475, -0.00000823375262 , -0.000005967671436],
     [ 0.000036781365687,  0.000042606324025,  0.00004963055636 ],
     [ 0.000006267830783, -0.000012114425726, -0.000026462669094],
     [ 0.00005598462845 , -0.000008233752626,  0.000005967681539],
     [-0.00003678136564 , -0.000042606324015,  0.00004963055628 ],
     [ 0.000030731202553,  0.00002570051094 ,  0.000056151104244],
     [-0.000014926915547, -0.000001173748799,  0.000005567152414],
     [-0.000200327181475, -0.000027280454112, -0.000071478160437]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001, normalization='intermediate')

    assert(np.max(np.abs(I-aat_ref)) < 1e-8)

def test_mp2_aat_full_norm():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    I_00_ref = np.array(
    [[-0.096723351176647, -0.024181272559026,  0.068430139624694],
     [ 0.02440026853828 ,  0.005811811081961, -0.00377502469012 ],
     [-0.206841614525112, -0.05120348217461 ,  0.092847829558901],
     [-0.087682713190163, -0.022005600141307,  0.059580490685184],
     [-0.016265639556674, -0.004009823395063,  0.02005712427525 ],
     [-0.212647898835307, -0.052555507198603,  0.087998129804366],
     [-0.087682713189233, -0.022005600140938, -0.059580490681089],
     [-0.016265639573135, -0.004009823398961, -0.02005712428386 ],
     [ 0.212647898839365,  0.052555507199591,  0.087998129794764],
     [-0.096723351213789, -0.024181272568481, -0.068430139622883],
     [ 0.024400268460648,  0.005811811062735,  0.003775024581204],
     [ 0.206841614521018,  0.051203482173443,  0.092847829566123]])

    I_0D_ref = np.array(
    [[ 0.006729870102103,  0.001675764841709, -0.005070693238922],
     [-0.000861858633735, -0.000212912387148, -0.000113861456894],
     [ 0.012919646614914,  0.003214061019705, -0.007987390964472],
     [ 0.006345853458475,  0.001580639671651, -0.004752827896155],
     [ 0.000967502144395,  0.000240579116881, -0.000539944722178],
     [ 0.013035012801975,  0.003242202768246, -0.00787402640068 ],
     [ 0.006345853458529,  0.001580639671666,  0.004752827895969],
     [ 0.000967502145657,  0.000240579117194,  0.000539944722927],
     [-0.013035012802303, -0.003242202768324, -0.007874026399958],
     [ 0.006729870104109,  0.001675764842218,  0.005070693238991],
     [-0.000861858628602, -0.000212912385862,  0.000113861470701],
     [-0.012919646614301, -0.003214061019542, -0.007987390965115]])

    I_D0_ref = np.array(
    [[-0.006729870187304, -0.001675764849652,  0.005070693251512],
     [ 0.000861858648603,  0.000212912388307,  0.000113861455361],
     [-0.012919646761546, -0.003214061032019,  0.007987391007058],
     [-0.006345853539045, -0.001580639681657,  0.004752827907825],
     [-0.000967502162469, -0.000240579120877,  0.000539944723795],
     [-0.013035012947752, -0.003242202781376,  0.007874026442289],
     [-0.006345853537955, -0.001580639681385, -0.004752827907489],
     [-0.000967502164103, -0.000240579121288, -0.000539944725327],
     [ 0.013035012947641,  0.00324220278135 ,  0.00787402644207 ],
     [-0.00672987018937 , -0.001675764850167, -0.005070693250025],
     [ 0.000861858644177,  0.000212912387211, -0.000113861468837],
     [ 0.012919646761641,  0.003214061032042,  0.007987391007655]])

    I_DD_ref = np.array(
    [[ 0.000030375220123,  0.000025402802803, -0.000055133014035],
     [-0.000014754005958, -0.000001160152387, -0.000005441973293],
     [ 0.000198006642816,  0.00002696444432 , -0.000069951395404],
     [ 0.000006195225855, -0.000011974095329,  0.000025758136646],
     [ 0.000055336116895, -0.000008138374957, -0.00000586253615 ],
     [ 0.000036355299728,  0.000042112783301,  0.000048356868496],
     [ 0.000006195225771, -0.000011974095351, -0.000025758136644],
     [ 0.000055336116907, -0.000008138374951,  0.000005862536224],
     [-0.0000363552997  , -0.000042112783298,  0.000048356868446],
     [ 0.000030375220166,  0.000025402802814,  0.000055133013892],
     [-0.000014754005978, -0.000001160152386,  0.000005441973575],
     [-0.000198006642872, -0.000026964444336, -0.000069951395413]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-8)

def test_cid_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting reference AAT.
    I_00_ref = np.array(
    [[ 0.000000000000615,  0.000000000000288, -0.226306484070573],
     [-0.000000000001284,  0.000000000002456,  0.000000000009745],
     [ 0.329611190264759, -0.000000000003012, -0.               ],
     [ 0.000000000000584, -0.000000000000022,  0.059895496710408],
     [ 0.000000000000922, -0.000000000000159, -0.136503781615856],
     [-0.229202569432311,  0.215872630800364,  0.000000000000086],
     [-0.000000000001647,  0.000000000001087,  0.05989549670396 ],
     [-0.000000000001775, -0.000000000000769,  0.136503781614182],
     [-0.229202569423767, -0.21587263080386 , -0.000000000000086]])

    I_0D_ref = np.array(
    [[-0.000000000000016, -0.000000000000004,  0.009719122239378],
     [-0.000000000000004, -0.00000000000003 , -0.000000000000007],
     [-0.008593312033004,  0.000000000000037, -0.               ],
     [-0.000000000000046,  0.               ,  0.001199562663158],
     [-0.000000000000058,  0.000000000000002,  0.00419309249157 ],
     [ 0.005975552929523, -0.002639002775059,  0.000000000000004],
     [ 0.000000000000074, -0.000000000000013,  0.001199562663249],
     [ 0.000000000000012,  0.000000000000009, -0.004193092491498],
     [ 0.005975552929327,  0.002639002775102, -0.000000000000004]])

    I_D0_ref = np.array(
    [[ 0.000000000000021, -0.000000000000008, -0.009719121979288],
     [ 0.000000000000077,  0.000000000000041,  0.000000000000138],
     [ 0.008593312029382, -0.000000000000037,  0.               ],
     [ 0.000000000000065,  0.000000000000007, -0.001199562713834],
     [ 0.000000000000082,  0.000000000000018, -0.004193092483275],
     [-0.005975553612676,  0.002639003322523, -0.000000000000003],
     [-0.000000000000075,  0.00000000000001 , -0.001199562713837],
     [ 0.000000000000094, -0.000000000000019,  0.004193092483204],
     [-0.005975553612565, -0.002639003322554,  0.000000000000003]])

    I_DD_ref = np.array(
    [[ 0.000000000000076,  0.000000000000007, -0.006571145180512],
     [ 0.000000000002677,  0.000000000001015, -0.000000000000175],
     [ 0.03605012126928 , -0.00000000000062 ,  0.000000000000001],
     [ 0.000000000000042, -0.000000000000022,  0.001267365504038],
     [ 0.000000000000008,  0.000000000000042, -0.009999089093697],
     [-0.020642645988524,  0.016765832724844, -0.000000000000034],
     [-0.000000000000056, -0.000000000000001,  0.001267365504546],
     [-0.000000000000143,  0.000000000000071,  0.009999089093786],
     [-0.020642645988882, -0.016765832725199,  0.000000000000003]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001, normalization='intermediate')

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

def test_cid_aat_full_norm():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    I_00_ref = np.array(
    [[ 0.000000000001681, -0.000000000000066, -0.216369026352184],
     [ 0.000000000031349,  0.000000000013539,  0.000000000004924],
     [ 0.315137467648931, -0.000000000006571,  0.000000000000019],
     [ 0.000000000001682, -0.00000000000047 ,  0.057265395478773],
     [ 0.000000000001331,  0.000000000000033, -0.130509695508135],
     [-0.21913794022654 ,  0.206393339272046, -0.000000000000701],
     [-0.000000000000366, -0.000000000000273,  0.057265395481647],
     [-0.000000000000126,  0.00000000000079 ,  0.130509695509671],
     [-0.219137940232017, -0.206393339277598, -0.000000000000611]])

    I_0D_ref = np.array(
    [[-0.000000000000044,  0.000000000000001,  0.009292340979379],
     [-0.000000000000829, -0.000000000000166,  0.000000000001179],
     [-0.008215966789908,  0.00000000000008 ,  0.               ],
     [-0.000000000000053,  0.000000000000006,  0.001146888064831],
     [-0.000000000000045, -0.               ,  0.00400896750085 ],
     [ 0.005713157422038, -0.002523120198581,  0.000000000000005],
     [ 0.000000000000019,  0.000000000000003,  0.001146888064285],
     [-0.000000000000007, -0.00000000000001 , -0.004008967500919],
     [ 0.005713157422109,  0.002523120198647, -0.000000000000005]])

    I_D0_ref = np.array(
    [[ 0.000000000000039, -0.000000000000023, -0.009292340730827],
     [ 0.00000000000088 ,  0.00000000000017 , -0.000000000001101],
     [ 0.00821596678659 , -0.000000000000105, -0.               ],
     [ 0.00000000000001 ,  0.000000000000015, -0.001146888112374],
     [ 0.000000000000005,  0.000000000000022, -0.004008967492092],
     [-0.005713158075148,  0.002523120721989, -0.000000000000004],
     [-0.000000000000055,  0.000000000000019, -0.001146888111887],
     [-0.000000000000021, -0.000000000000001,  0.004008967492141],
     [-0.005713158075265, -0.002523120722027,  0.000000000000004]])

    I_DD_ref = np.array(
    [[ 0.000000000000072,  0.000000000000007, -0.006282596323029],
     [ 0.000000000002559,  0.00000000000097 , -0.000000000000168],
     [ 0.034467106277752, -0.000000000000593,  0.000000000000001],
     [ 0.00000000000004 , -0.000000000000021,  0.001211713581405],
     [ 0.000000000000008,  0.00000000000004 , -0.009560014070841],
     [-0.019736196388032,  0.016029619821933, -0.000000000000033],
     [-0.000000000000054, -0.000000000000001,  0.001211713581891],
     [-0.000000000000137,  0.000000000000068,  0.009560014070926],
     [-0.019736196388375, -0.016029619822273,  0.000000000000003]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_mp2_SO_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'MP2_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference AAT.
    I_00_ref = np.array(
    [[-0.097856900360615, -0.024464664951138,  0.069232106541966],
     [ 0.024686227422983,  0.005879922586945, -0.003819266089423],
     [-0.209265694313652, -0.051803561259253,  0.093935959595289],
     [-0.088710310616063, -0.022263494739757,  0.060278744166398],
     [-0.016456264690831, -0.004056816513517,  0.020292183723423],
     [-0.215140025600313, -0.05317143139671 ,  0.089029423759874],
     [-0.088710310631667, -0.022263494743767, -0.060278744172876],
     [-0.016456264697269, -0.004056816515104, -0.020292184243793],
     [ 0.215140025582485,  0.053171431392279,  0.089029423753599],
     [-0.097856900385283, -0.02446466495731 , -0.069232106543105],
     [ 0.024686227444831,  0.005879922592439,  0.00381926608413 ],
     [ 0.209265694510914,  0.051803561307904,  0.093935959578609]])
    
    I_0D_ref = np.array(
    [[ 0.006808740805464,  0.001695403965524, -0.005130119220599],
     [-0.000871959182927, -0.00021540761311 , -0.000115195855897],
     [ 0.013071058381498,  0.003251728200012, -0.008080999183209],
     [ 0.006420223678521,  0.001599163975967, -0.004808528656272],
     [ 0.00097884078409 ,  0.00024339858343 , -0.00054627260451 ],
     [ 0.013187776617029,  0.003280199759361, -0.007966306041871],
     [ 0.006420223679836,  0.001599163976289,  0.004808528656924],
     [ 0.000978840784731,  0.0002433985836  ,  0.000546272620396],
     [-0.013187776615873, -0.003280199759082, -0.007966306041163],
     [ 0.006808740806848,  0.001695403965874,  0.005130119220869],
     [-0.000871959184344, -0.000215407613452,  0.000115195856348],
     [-0.013071058393264, -0.003251728202942, -0.008080999183206]])
    
    I_D0_ref = np.array(
    [[-0.006808740892117, -0.001695403973672,  0.005130119233505],
     [ 0.0008719591981  ,  0.000215407614326,  0.000115195854473],
     [-0.013071058531117, -0.003251728212792,  0.008080999226863],
     [-0.006420223760504, -0.001599163986204,  0.004808528668143],
     [-0.000978840803084, -0.000243398587652,  0.000546272606263],
     [-0.01318777676535 , -0.003280199772866,  0.007966306084481],
     [-0.006420223761249, -0.001599163986388, -0.004808528668831],
     [-0.000978840803297, -0.000243398587705, -0.000546272622777],
     [ 0.013187776763665,  0.003280199772445,  0.007966306084511],
     [-0.006808740893785, -0.001695403974086, -0.005130119232304],
     [ 0.000871959199832,  0.000215407614753, -0.000115195854557],
     [ 0.013071058542963,  0.003251728215738,  0.008080999227038]])

    I_DD_ref = np.array(
    [[ 0.000030731202513,  0.000025700510925, -0.000056151104364],
     [-0.000014926915508, -0.000001173748787, -0.000005567152441],
     [ 0.000200327181323,  0.000027280454078, -0.000071478160397],
     [ 0.000006267830815, -0.000012114425716,  0.000026462669104],
     [ 0.000055984628475, -0.00000823375262 , -0.000005967671436],
     [ 0.000036781365687,  0.000042606324025,  0.00004963055636 ],
     [ 0.000006267830783, -0.000012114425726, -0.000026462669094],
     [ 0.00005598462845 , -0.000008233752626,  0.000005967681539],
     [-0.00003678136564 , -0.000042606324015,  0.00004963055628 ],
     [ 0.000030731202553,  0.00002570051094 ,  0.000056151104244],
     [-0.000014926915547, -0.000001173748799,  0.000005567152414],
     [-0.000200327181475, -0.000027280454112, -0.000071478160437]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001, normalization='intermediate')

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_mp2_SO_aat_full_norm():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'MP2_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    I_00_ref = np.array(
    [[-0.096723351176647, -0.024181272559026,  0.068430139624694],
     [ 0.02440026853828 ,  0.005811811081961, -0.00377502469012 ],
     [-0.206841614525112, -0.05120348217461 ,  0.092847829558901],
     [-0.087682713190163, -0.022005600141307,  0.059580490685184],
     [-0.016265639556674, -0.004009823395063,  0.02005712427525 ],
     [-0.212647898835307, -0.052555507198603,  0.087998129804366],
     [-0.087682713189233, -0.022005600140938, -0.059580490681089],
     [-0.016265639573135, -0.004009823398961, -0.02005712428386 ],
     [ 0.212647898839365,  0.052555507199591,  0.087998129794764],
     [-0.096723351213789, -0.024181272568481, -0.068430139622883],
     [ 0.024400268460648,  0.005811811062735,  0.003775024581204],
     [ 0.206841614521018,  0.051203482173443,  0.092847829566123]])

    I_0D_ref = np.array(
    [[ 0.006729870102103,  0.001675764841709, -0.005070693238922],
     [-0.000861858633735, -0.000212912387148, -0.000113861456894],
     [ 0.012919646614914,  0.003214061019705, -0.007987390964472],
     [ 0.006345853458475,  0.001580639671651, -0.004752827896155],
     [ 0.000967502144395,  0.000240579116881, -0.000539944722178],
     [ 0.013035012801975,  0.003242202768246, -0.00787402640068 ],
     [ 0.006345853458529,  0.001580639671666,  0.004752827895969],
     [ 0.000967502145657,  0.000240579117194,  0.000539944722927],
     [-0.013035012802303, -0.003242202768324, -0.007874026399958],
     [ 0.006729870104109,  0.001675764842218,  0.005070693238991],
     [-0.000861858628602, -0.000212912385862,  0.000113861470701],
     [-0.012919646614301, -0.003214061019542, -0.007987390965115]])

    I_D0_ref = np.array(
    [[-0.006729870187304, -0.001675764849652,  0.005070693251512],
     [ 0.000861858648603,  0.000212912388307,  0.000113861455361],
     [-0.012919646761546, -0.003214061032019,  0.007987391007058],
     [-0.006345853539045, -0.001580639681657,  0.004752827907825],
     [-0.000967502162469, -0.000240579120877,  0.000539944723795],
     [-0.013035012947752, -0.003242202781376,  0.007874026442289],
     [-0.006345853537955, -0.001580639681385, -0.004752827907489],
     [-0.000967502164103, -0.000240579121288, -0.000539944725327],
     [ 0.013035012947641,  0.00324220278135 ,  0.00787402644207 ],
     [-0.00672987018937 , -0.001675764850167, -0.005070693250025],
     [ 0.000861858644177,  0.000212912387211, -0.000113861468837],
     [ 0.012919646761641,  0.003214061032042,  0.007987391007655]])

    I_DD_ref = np.array(
    [[ 0.000030375220123,  0.000025402802803, -0.000055133014035],
     [-0.000014754005958, -0.000001160152387, -0.000005441973293],
     [ 0.000198006642816,  0.00002696444432 , -0.000069951395404],
     [ 0.000006195225855, -0.000011974095329,  0.000025758136646],
     [ 0.000055336116895, -0.000008138374957, -0.00000586253615 ],
     [ 0.000036355299728,  0.000042112783301,  0.000048356868496],
     [ 0.000006195225771, -0.000011974095351, -0.000025758136644],
     [ 0.000055336116907, -0.000008138374951,  0.000005862536224],
     [-0.0000363552997  , -0.000042112783298,  0.000048356868446],
     [ 0.000030375220166,  0.000025402802814,  0.000055133013892],
     [-0.000014754005978, -0.000001160152386,  0.000005441973575],
     [-0.000198006642872, -0.000026964444336, -0.000069951395413]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_cid_SO_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'CID_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting reference AAT.
    I_00_ref = np.array(
    [[-0.097856900379267, -0.024464664955929,  0.06923210655395 ],
     [ 0.024686227457246,  0.005879922595279, -0.003819266076075],
     [-0.209265694502983, -0.051803561305989,  0.093935959571669],
     [-0.088710310622527, -0.022263494741675,  0.060278744164669],
     [-0.016456264699243, -0.004056816515349,  0.020292183724675],
     [-0.215140025586322, -0.053171431393181,  0.089029423759699],
     [-0.088710310630709, -0.022263494743629, -0.060278744175718],
     [-0.016456264688766, -0.004056816512675, -0.020292183730881],
     [ 0.215140025581244,  0.053171431391934,  0.089029423752232],
     [-0.097856900442317, -0.024464664971436, -0.069232106526598],
     [ 0.024686227479546,  0.005879922601014,  0.003819266091157],
     [ 0.209265694494609,  0.05180356130386 ,  0.093935959561269]])

    I_0D_ref = np.array(
    [[ 0.011424906174176,  0.00284279020091 , -0.007657421068027],
     [-0.001518637652337, -0.00037607040392 , -0.000158476957254],
     [ 0.022030689055581,  0.005478587577683, -0.012024128553204],
     [ 0.010756711930063,  0.002677055982711, -0.007165839091218],
     [ 0.001653378746906,  0.000411045954502, -0.000847771162833],
     [ 0.022245707233434,  0.005531466675376, -0.011844514710824],
     [ 0.010756711931167,  0.002677055982973,  0.007165839092493],
     [ 0.001653378745958,  0.000411045954255,  0.00084777116338 ],
     [-0.022245707232918, -0.005531466675226, -0.011844514709802],
     [ 0.011424906180991,  0.002842790202584,  0.007657421064533],
     [-0.001518637654643, -0.000376070404534,  0.00015847695525 ],
     [-0.022030689054449, -0.005478587577412, -0.012024128552076]])

    I_D0_ref = np.array(
    [[-0.011424906361104, -0.002842790219884,  0.007657421111043],
     [ 0.001518637672543,  0.000376070404646,  0.000158476956634],
     [-0.022030689406287, -0.005478587615383,  0.012024128645684],
     [-0.010756712106327, -0.002677056005323,  0.007165839136035],
     [-0.00165337876797 , -0.000411045958558,  0.000847771170003],
     [-0.022245707576422, -0.005531466713657,  0.011844514802773],
     [-0.010756712108056, -0.002677056005755, -0.007165839137041],
     [-0.001653378767336, -0.000411045958402, -0.000847771170318],
     [ 0.022245707576832,  0.005531466713759,  0.011844514803097],
     [-0.011424906367372, -0.002842790221441, -0.007657421106823],
     [ 0.001518637675001,  0.000376070405261, -0.000158476955133],
     [ 0.022030689404459,  0.005478587614932,  0.012024128644922]])

    I_DD_ref = np.array(
    [[ 0.000124891106513,  0.000087456328189, -0.000161939000276],
     [-0.00008369903897 , -0.000020174481159, -0.000020471011177],
     [ 0.000683239104114,  0.000075313802282, -0.000172953608345],
     [ 0.00001423873044 , -0.000032832367992,  0.000091401848996],
     [ 0.000178358493551, -0.00004344393336 ,  0.000006134045722],
     [ 0.000156210348074,  0.000172437039597,  0.000118608684893],
     [ 0.000014238749811, -0.000032832367697, -0.000091401848852],
     [ 0.000178358494032, -0.000043443936217, -0.000006134045661],
     [-0.000156210378479, -0.000172437041579,  0.000118608684902],
     [ 0.000124891089301,  0.000087456327671,  0.000161938999836],
     [-0.000083699043171, -0.000020174480659,  0.00002047101123 ],
     [-0.000683239073748, -0.00007531380032 , -0.00017295360844 ]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001, normalization='intermediate')

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_cid_SO_aat_full_norm():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'CID_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    I_00_ref = np.array(
    [[-0.09513701717291 , -0.023784681925721,  0.067307835069183],
     [ 0.024000086211862,  0.005716493110661, -0.003713111511762],
     [-0.203449260104919, -0.050363707449073,  0.09132505695627 ],
     [-0.086244652286479, -0.021644692132782,  0.058603326872811],
     [-0.015998871128165, -0.003944059347583,  0.01972817270957 ],
     [-0.209160317098023, -0.051693558272251,  0.086554895833264],
     [-0.08624465231163 , -0.021644692138982, -0.058603326884424],
     [-0.015998871121655, -0.003944059345944, -0.019728172712171],
     [ 0.209160317113225,  0.051693558276127,  0.086554895820963],
     [-0.095137017212308, -0.023784681935333, -0.067307835065207],
     [ 0.024000086213199,  0.005716493111224,  0.003713111512817],
     [ 0.203449260102882,  0.050363707448667,  0.091325056947525]])

    I_0D_ref = np.array(
    [[ 0.011107356668396,  0.002763776280464, -0.007444586911469],
     [-0.001476427884344, -0.000365617716593, -0.00015407216886 ],
     [ 0.021418357168675,  0.00532631299945 , -0.011689923965106],
     [ 0.010457734547885,  0.002602648561722, -0.006966668207858],
     [ 0.001607423919157,  0.000399621139614, -0.000824207790605],
     [ 0.02162739902239 ,  0.005377722348694, -0.011515302398893],
     [ 0.010457734550992,  0.002602648562475,  0.006966668209321],
     [ 0.001607423918804,  0.000399621139512,  0.000824207790541],
     [-0.02162739902415 , -0.005377722349145, -0.011515302397462],
     [ 0.01110735667196 ,  0.002763776281365,  0.007444586911305],
     [-0.001476427884461, -0.000365617716593,  0.000154072168972],
     [-0.021418357168174, -0.005326312999309, -0.011689923963874]])

    I_D0_ref = np.array(
    [[-0.011107356848449, -0.002763776298499,  0.007444586952933],
     [ 0.001476427904045,  0.000365617717291,  0.000154072168612],
     [-0.021418357503999, -0.005326313034694,  0.011689924052147],
     [-0.010457734718616, -0.002602648583527,  0.0069666682508  ],
     [-0.001607423940173, -0.000399621143692,  0.000824207797632],
     [-0.021627399353107, -0.005377722385237,  0.011515302485561],
     [-0.010457734720092, -0.002602648583895, -0.006966668251859],
     [-0.001607423939574, -0.000399621143549, -0.00082420779848 ],
     [ 0.021627399353798,  0.005377722385406,  0.011515302484935],
     [-0.011107356852376, -0.002763776299474, -0.007444586950971],
     [ 0.001476427905098,  0.000365617717557, -0.000154072168121],
     [ 0.021418357503382,  0.005326313034541,  0.011689924051268]])

    I_DD_ref = np.array(
    [[ 0.00012141981371 ,  0.000085025523518, -0.000154934043171],
     [-0.00008137266825 , -0.000019613741413, -0.000019434710879],
     [ 0.000664248804108,  0.000073220492216, -0.000163554347185],
     [ 0.000013842980189, -0.000031919808921,  0.000086123777585],
     [ 0.000173401108829, -0.000042236432859,  0.00000619635474 ],
     [ 0.000151868572951,  0.000167644239681,  0.000110719913945],
     [ 0.000013842979618, -0.000031919810028, -0.000086123777912],
     [ 0.000173401110031, -0.000042236432248, -0.000006196354662],
     [-0.000151868570934, -0.000167644237522,  0.000110719913684],
     [ 0.000121419814438,  0.000085025524589,  0.000154934043438],
     [-0.000081372668693, -0.00001961374137 ,  0.000019434711171],
     [-0.000664248806083, -0.000073220494365, -0.000163554347132]])

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Not ready yet.")
def test_cisd_SO_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    # Setting reference AAT.
    I_00_ref = np.array(
    [[-0.097856900379267, -0.024464664955929,  0.06923210655395 ],
     [ 0.024686227457246,  0.005879922595279, -0.003819266076075],
     [-0.209265694502983, -0.051803561305989,  0.093935959571669],
     [-0.088710310622527, -0.022263494741675,  0.060278744164669],
     [-0.016456264699243, -0.004056816515349,  0.020292183724675],
     [-0.215140025586322, -0.053171431393181,  0.089029423759699],
     [-0.088710310630709, -0.022263494743629, -0.060278744175718],
     [-0.016456264688766, -0.004056816512675, -0.020292183730881],
     [ 0.215140025581244,  0.053171431391934,  0.089029423752232],
     [-0.097856900442317, -0.024464664971436, -0.069232106526598],
     [ 0.024686227479546,  0.005879922601014,  0.003819266091157],
     [ 0.209265694494609,  0.05180356130386 ,  0.093935959561269]])

    I_0D_ref = np.array(
    [[ 0.011424906174176,  0.00284279020091 , -0.007657421068027],
     [-0.001518637652337, -0.00037607040392 , -0.000158476957254],
     [ 0.022030689055581,  0.005478587577683, -0.012024128553204],
     [ 0.010756711930063,  0.002677055982711, -0.007165839091218],
     [ 0.001653378746906,  0.000411045954502, -0.000847771162833],
     [ 0.022245707233434,  0.005531466675376, -0.011844514710824],
     [ 0.010756711931167,  0.002677055982973,  0.007165839092493],
     [ 0.001653378745958,  0.000411045954255,  0.00084777116338 ],
     [-0.022245707232918, -0.005531466675226, -0.011844514709802],
     [ 0.011424906180991,  0.002842790202584,  0.007657421064533],
     [-0.001518637654643, -0.000376070404534,  0.00015847695525 ],
     [-0.022030689054449, -0.005478587577412, -0.012024128552076]])

    I_D0_ref = np.array(
    [[-0.011424906361104, -0.002842790219884,  0.007657421111043],
     [ 0.001518637672543,  0.000376070404646,  0.000158476956634],
     [-0.022030689406287, -0.005478587615383,  0.012024128645684],
     [-0.010756712106327, -0.002677056005323,  0.007165839136035],
     [-0.00165337876797 , -0.000411045958558,  0.000847771170003],
     [-0.022245707576422, -0.005531466713657,  0.011844514802773],
     [-0.010756712108056, -0.002677056005755, -0.007165839137041],
     [-0.001653378767336, -0.000411045958402, -0.000847771170318],
     [ 0.022245707576832,  0.005531466713759,  0.011844514803097],
     [-0.011424906367372, -0.002842790221441, -0.007657421106823],
     [ 0.001518637675001,  0.000376070405261, -0.000158476955133],
     [ 0.022030689404459,  0.005478587614932,  0.012024128644922]])

    I_DD_ref = np.array(
    [[ 0.000124891106513,  0.000087456328189, -0.000161939000276],
     [-0.00008369903897 , -0.000020174481159, -0.000020471011177],
     [ 0.000683239104114,  0.000075313802282, -0.000172953608345],
     [ 0.00001423873044 , -0.000032832367992,  0.000091401848996],
     [ 0.000178358493551, -0.00004344393336 ,  0.000006134045722],
     [ 0.000156210348074,  0.000172437039597,  0.000118608684893],
     [ 0.000014238749811, -0.000032832367697, -0.000091401848852],
     [ 0.000178358494032, -0.000043443936217, -0.000006134045661],
     [-0.000156210378479, -0.000172437041579,  0.000118608684902],
     [ 0.000124891089301,  0.000087456327671,  0.000161938999836],
     [-0.000083699043171, -0.000020174480659,  0.00002047101123 ],
     [-0.000683239073748, -0.00007531380032 , -0.00017295360844 ]]) 

    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref

    # Compute AATs in parallel.
    I = apyib.parallel.compute_parallel_aats(parameters, 0.0001, 0.0001, normalization='intermediate')
    print(aat_ref)

    assert(np.max(np.abs(I-aat_ref)) < 1e-7)

