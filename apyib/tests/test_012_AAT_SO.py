import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

#@pytest.mark.skip(reason="Too slow.")
def test_rhf_SO_aat():
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

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute finite difference AATs inputs.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.0001, 0.0001)

    # Compute finite difference AATs.
    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.0001, 0.0001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta)

    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

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

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute finite difference AATs inputs.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.0001, 0.0001)

    # Compute finite difference AATs.
    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.0001, 0.0001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta, normalization='intermediate')

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

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

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute finite difference AATs inputs.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.0001, 0.0001)

    # Compute finite difference AATs.
    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.0001, 0.0001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    #I_00 = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    #I_0D = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    #I_D0 = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    #I_DD = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta)
            #I_00[lambd_alpha][beta], I_0D[lambd_alpha][beta], I_D0[lambd_alpha][beta], I_DD[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta)

    #aat = I_00 + I_0D + I_D0 + I_DD    

    #print("I_00: \n", I_00, "\n")
    #print("I_0D: \n", I_0D, "\n")
    #print("I_D0: \n", I_D0, "\n")
    #print("I_DD: \n", I_DD, "\n")

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

#@pytest.mark.skip(reason="Too slow.")
#def test_cid_SO_aat():
#    # Set parameters for the calculation.
#    parameters = {'geom': moldict["H2O"],
#                  'basis': 'STO-6G',
#                  'method': 'CID_SO',
#                  'e_convergence': 1e-12,
#                  'd_convergence': 1e-12,
#                  'DIIS': True,
#                  'freeze_core': False,
#                  'F_el': [0.0, 0.0, 0.0],
#                  'F_mag': [0.0, 0.0, 0.0],
#                  'max_iterations': 120}
#
#    # Setting reference AAT.
#    I_00_ref = np.array(
#    [[ 0.000000000000615,  0.000000000000288, -0.226306484070573],
#     [-0.000000000001284,  0.000000000002456,  0.000000000009745],
#     [ 0.329611190264759, -0.000000000003012, -0.               ],
#     [ 0.000000000000584, -0.000000000000022,  0.059895496710408],
#     [ 0.000000000000922, -0.000000000000159, -0.136503781615856],
#     [-0.229202569432311,  0.215872630800364,  0.000000000000086],
#     [-0.000000000001647,  0.000000000001087,  0.05989549670396 ],
#     [-0.000000000001775, -0.000000000000769,  0.136503781614182],
#     [-0.229202569423767, -0.21587263080386 , -0.000000000000086]])
#
#    I_0D_ref = np.array(
#    [[-0.000000000000016, -0.000000000000004,  0.009719122239378],
#     [-0.000000000000004, -0.00000000000003 , -0.000000000000007],
#     [-0.008593312033004,  0.000000000000037, -0.               ],
#     [-0.000000000000046,  0.               ,  0.001199562663158],
#     [-0.000000000000058,  0.000000000000002,  0.00419309249157 ],
#     [ 0.005975552929523, -0.002639002775059,  0.000000000000004],
#     [ 0.000000000000074, -0.000000000000013,  0.001199562663249],
#     [ 0.000000000000012,  0.000000000000009, -0.004193092491498],
#     [ 0.005975552929327,  0.002639002775102, -0.000000000000004]])
#
#    I_D0_ref = np.array(
#    [[ 0.000000000000021, -0.000000000000008, -0.009719121979288],
#     [ 0.000000000000077,  0.000000000000041,  0.000000000000138],
#     [ 0.008593312029382, -0.000000000000037,  0.               ],
#     [ 0.000000000000065,  0.000000000000007, -0.001199562713834],
#     [ 0.000000000000082,  0.000000000000018, -0.004193092483275],
#     [-0.005975553612676,  0.002639003322523, -0.000000000000003],
#     [-0.000000000000075,  0.00000000000001 , -0.001199562713837],
#     [ 0.000000000000094, -0.000000000000019,  0.004193092483204],
#     [-0.005975553612565, -0.002639003322554,  0.000000000000003]])
#
#    I_DD_ref = np.array(
#    [[ 0.000000000000076,  0.000000000000007, -0.006571145180512],
#     [ 0.000000000002677,  0.000000000001015, -0.000000000000175],
#     [ 0.03605012126928 , -0.00000000000062 ,  0.000000000000001],
#     [ 0.000000000000042, -0.000000000000022,  0.001267365504038],
#     [ 0.000000000000008,  0.000000000000042, -0.009999089093697],
#     [-0.020642645988524,  0.016765832724844, -0.000000000000034],
#     [-0.000000000000056, -0.000000000000001,  0.001267365504546],
#     [-0.000000000000143,  0.000000000000071,  0.009999089093786],
#     [-0.020642645988882, -0.016765832725199,  0.000000000000003]])
#
#    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref
#
#    # Compute energy.
#    E_list, T_list, C, basis = apyib.energy.energy(parameters)
#    E_tot = E_list[0] + E_list[1] + E_list[2]
#    print(E_tot)
#
#    H = apyib.hamiltonian.Hamiltonian(parameters)
#    wfn = apyib.hf_wfn.hf_wfn(H)
#
#    # Compute finite difference AATs inputs.
#    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
#    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.0001, 0.0001)
#
#    # Compute finite difference AATs.
#    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.0001, 0.0001)
#    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
#    for lambd_alpha in range(3 * H.molecule.natom()):
#        for beta in range(3):
#            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta, normalization='intermediate')
#
#    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

#@pytest.mark.skip(reason="Too slow.")
#def test_cid_SO_aat_full_norm():
#    # Set parameters for the calculation.
#    parameters = {'geom': moldict["H2O"],
#                  'basis': 'STO-6G',
#                  'method': 'CID_SO',
#                  'e_convergence': 1e-12,
#                  'd_convergence': 1e-12,
#                  'DIIS': True,
#                  'freeze_core': False,
#                  'F_el': [0.0, 0.0, 0.0],
#                  'F_mag': [0.0, 0.0, 0.0],
#                  'max_iterations': 120}
#
#    # Setting fully normalized reference AAT.
#    I_00_ref = np.array(
#    [[ 0.000000000001681, -0.000000000000066, -0.216369026352184],
#     [ 0.000000000031349,  0.000000000013539,  0.000000000004924],
#     [ 0.315137467648931, -0.000000000006571,  0.000000000000019],
#     [ 0.000000000001682, -0.00000000000047 ,  0.057265395478773],
#     [ 0.000000000001331,  0.000000000000033, -0.130509695508135],
#     [-0.21913794022654 ,  0.206393339272046, -0.000000000000701],
#     [-0.000000000000366, -0.000000000000273,  0.057265395481647],
#     [-0.000000000000126,  0.00000000000079 ,  0.130509695509671],
#     [-0.219137940232017, -0.206393339277598, -0.000000000000611]])
#
#    I_0D_ref = np.array(
#    [[-0.000000000000044,  0.000000000000001,  0.009292340979379],
#     [-0.000000000000829, -0.000000000000166,  0.000000000001179],
#     [-0.008215966789908,  0.00000000000008 ,  0.               ],
#     [-0.000000000000053,  0.000000000000006,  0.001146888064831],
#     [-0.000000000000045, -0.               ,  0.00400896750085 ],
#     [ 0.005713157422038, -0.002523120198581,  0.000000000000005],
#     [ 0.000000000000019,  0.000000000000003,  0.001146888064285],
#     [-0.000000000000007, -0.00000000000001 , -0.004008967500919],
#     [ 0.005713157422109,  0.002523120198647, -0.000000000000005]])
#
#    I_D0_ref = np.array(
#    [[ 0.000000000000039, -0.000000000000023, -0.009292340730827],
#     [ 0.00000000000088 ,  0.00000000000017 , -0.000000000001101],
#     [ 0.00821596678659 , -0.000000000000105, -0.               ],
#     [ 0.00000000000001 ,  0.000000000000015, -0.001146888112374],
#     [ 0.000000000000005,  0.000000000000022, -0.004008967492092],
#     [-0.005713158075148,  0.002523120721989, -0.000000000000004],
#     [-0.000000000000055,  0.000000000000019, -0.001146888111887],
#     [-0.000000000000021, -0.000000000000001,  0.004008967492141],
#     [-0.005713158075265, -0.002523120722027,  0.000000000000004]])
#
#    I_DD_ref = np.array(
#    [[ 0.000000000000072,  0.000000000000007, -0.006282596323029],
#     [ 0.000000000002559,  0.00000000000097 , -0.000000000000168],
#     [ 0.034467106277752, -0.000000000000593,  0.000000000000001],
#     [ 0.00000000000004 , -0.000000000000021,  0.001211713581405],
#     [ 0.000000000000008,  0.00000000000004 , -0.009560014070841],
#     [-0.019736196388032,  0.016029619821933, -0.000000000000033],
#     [-0.000000000000054, -0.000000000000001,  0.001211713581891],
#     [-0.000000000000137,  0.000000000000068,  0.009560014070926],
#     [-0.019736196388375, -0.016029619822273,  0.000000000000003]])
#
#    aat_ref = I_00_ref + I_D0_ref + I_0D_ref + I_DD_ref
#
#    # Compute energy.
#    E_list, T_list, C, basis = apyib.energy.energy(parameters)
#    E_tot = E_list[0] + E_list[1] + E_list[2]
#    print(E_tot)
#
#    H = apyib.hamiltonian.Hamiltonian(parameters)
#    wfn = apyib.hf_wfn.hf_wfn(H)
#
#    # Compute finite difference AATs inputs.
#    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
#    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.0001, 0.0001)
#
#    # Compute finite difference AATs.
#    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.0001, 0.0001)
#    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
#    for lambd_alpha in range(3 * H.molecule.natom()):
#        for beta in range(3):
#            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta)
#
#    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_cid_SO_aat_full_norm():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-3G',
                  'method': 'CID_SO',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    aat_ref = np.array(
    [[-0.095540814320978, -0.023838374184792,  0.067534308328104],
     [ 0.024859894690783,  0.005930943259976, -0.00402322935501 ],
     [-0.203998603555246, -0.050615735214713,  0.091451278012954],
     [-0.086426692706765, -0.021733317747402,  0.05884087885991 ],
     [-0.016994525637844, -0.004269530902621,  0.020306187975289],
     [-0.21037648261415 , -0.051889007431137,  0.086924374842542],
     [-0.086426693169612, -0.02173331316276 , -0.058840875550237],
     [-0.016994530646993, -0.004269531583851, -0.020306190977836],
     [ 0.210376480592374,  0.051888999607868,  0.086924255327167],
     [-0.095540814399817, -0.023838378409075, -0.067534324183816],
     [ 0.024859895761856,  0.005930942535482,  0.004023230328119],
     [ 0.203998605580878,  0.050615743039658,  0.091451279747092]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute finite difference AATs inputs.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.000001, 0.000001)

    # Compute finite difference AATs.
    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.000001, 0.000001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_cisd_SO_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-3G',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    aat_ref = np.array(
    [[-0.0959055286, -0.02391999  ,  0.0674080955],
     [ 0.0191224523,  0.0045302271, -0.0026461627],
     [-0.2037785253, -0.0505421735,  0.0929373342],
     [-0.0896531257, -0.0225174447,  0.0604971932],
     [-0.0115279984, -0.0029403899,  0.0180324154],
     [-0.2104465742, -0.0518826005,  0.0894213555],
     [-0.0896531255, -0.0225174447, -0.0604971936],
     [-0.011527999 , -0.0029403901, -0.0180324155],
     [ 0.2104465742,  0.0518826006,  0.0894213555],
     [-0.0959055288, -0.0239199901, -0.0674080953],
     [ 0.0191224532,  0.0045302273,  0.0026461629],
     [ 0.2037785259,  0.0505421736,  0.0929373346]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute finite difference AATs inputs.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.000001, 0.000001)

    # Compute finite difference AATs.
    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.000001, 0.000001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta, normalization='intermediate')

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

@pytest.mark.skip(reason="Too slow.")
def test_cisd_SO_aat_full_norm():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-3G',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting fully normalized reference AAT.
    aat_ref = np.array(
    [[-0.0932435402, -0.0232560581,  0.0655477146],
     [ 0.0185916827,  0.0044044845, -0.0025707274],
     [-0.198122375 , -0.0491393066,  0.0903772041],
     [-0.0871646814, -0.0218924424,  0.0588063949],
     [-0.0112080231, -0.0028587754,  0.0175329454],
     [-0.2046053431, -0.0504425284,  0.0869198719],
     [-0.087164681 , -0.0218924424, -0.0588063955],
     [-0.0112080237, -0.0028587756, -0.0175329455],
     [ 0.2046053431,  0.0504425284,  0.0869198719],
     [-0.0932435404, -0.0232560582, -0.0655477145],
     [ 0.0185916834,  0.0044044847,  0.0025707274],
     [ 0.1981223755,  0.0491393068,  0.0903772046]]) 

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute finite difference AATs inputs.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = finite_difference.compute_AAT(0.000001, 0.000001)

    # Compute finite difference AATs.
    AATs = apyib.aats.AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, 0.000001, 0.000001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_SO_aats(lambd_alpha, beta)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)
