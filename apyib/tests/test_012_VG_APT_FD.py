import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *


def test_rhf_vg_apt():
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 6.743616471627435e+00, -6.756986186365436e-11,  7.832063220955714e-13],
     [-4.018186511517134e-11,  7.397291754065713e+00,  2.861150047311786e-12],
     [-3.972699169530089e-14, -2.533114408198812e-15,  8.298847296853211e+00],
     [ 5.087710943149175e-01, -4.146033992074074e-01,  5.693830826608728e-13],
     [-4.159634262254959e-01,  5.127757512886825e-01,  6.462243721049118e-13],
     [-8.395550759873335e-14, -3.313684574632568e-14,  7.921897971562962e-01],
     [ 5.087710943551254e-01,  4.146033992238530e-01, -7.037489383592156e-13],
     [ 4.159634262571429e-01,  5.127757512159606e-01, -3.686945247916614e-13],
     [ 1.201479612541332e-13,  3.468740297372424e-14,  7.921897971563998e-01]])

    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T) = \
        finite_difference.compute_VG_APT(0.0001, 0.0001)

    VG_APTs = apyib.vg_apts_fd.VG_APT(parameters, wfn, C, basis, T_list,
        nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
        0.0001, 0.0001)
    vg_apt = np.zeros((3 * H.molecule.natom(), 3))
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            vg_apt[lambd_alpha][beta] = VG_APTs.compute_spatial_vg_apts(lambd_alpha, beta)

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-7)


def test_mp2_vg_apt():
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 8.572396770530035e-01, -7.559430868258476e-04, -2.642168437697568e-01],
     [ 1.611207360840840e-02,  9.920446590886923e-01, -6.822165921978578e-03],
     [-2.700847435341381e-01,  1.226343131510618e-02,  5.812264686321112e-01],
     [ 8.646843766547287e-01, -3.168781106529372e-03, -2.486722511965976e-01],
     [-2.002206176996521e-02,  1.001210721896598e+00, -2.593981450432889e-02],
     [-2.719394507732442e-01,  1.494149156158541e-02,  5.861936044684426e-01],
     [ 8.646843766372080e-01, -3.168781103904152e-03,  2.486722512510167e-01],
     [-2.002206177920077e-02,  1.001210721897973e+00,  2.593981451158348e-02],
     [ 2.719394507761185e-01, -1.494149156400778e-02,  5.861936044403562e-01],
     [ 8.572396770305324e-01, -7.559430889218474e-04,  2.642168437336123e-01],
     [ 1.611207361557041e-02,  9.920446590869469e-01,  6.822165910867008e-03],
     [ 2.700847434544701e-01, -1.226343130344240e-02,  5.812264689891309e-01]])

    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T) = \
        finite_difference.compute_VG_APT(0.0001, 0.0001)

    VG_APTs = apyib.vg_apts_fd.VG_APT(parameters, wfn, C, basis, T_list,
        nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
        0.0001, 0.0001)
    vg_apt = np.zeros((3 * H.molecule.natom(), 3))
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            vg_apt[lambd_alpha][beta] = VG_APTs.compute_spatial_vg_apts(lambd_alpha, beta, normalization='intermediate')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-7)


def test_mp2_vg_apt_full_norm():
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 8.588933759284846e-01, -7.471864363987897e-04, -2.611572889862127e-01],
     [ 1.592543548965695e-02,  9.921368117126759e-01, -6.743315545857633e-03],
     [-2.669561513063203e-01,  1.212137487927043e-02,  5.860754089583907e-01],
     [ 8.662518380440263e-01, -3.132074760792831e-03, -2.457905426708449e-01],
     [-1.979013135378825e-02,  1.001196697205394e+00, -2.563943868650645e-02],
     [-2.687893740934226e-01,  1.476841316435276e-02,  5.909890559215814e-01],
     [ 8.662518380267086e-01, -3.132074758198009e-03,  2.457905427246329e-01],
     [-1.979013136291682e-02,  1.001196697206753e+00,  2.563943869367769e-02],
     [ 2.687893740962636e-01, -1.476841316674711e-02,  5.909890558938196e-01],
     [ 8.588933759062738e-01, -7.471864384705286e-04,  2.611572889504862e-01],
     [ 1.592543549673600e-02,  9.921368117109507e-01,  6.743315534874257e-03],
     [ 2.669561512275753e-01, -1.212137486774174e-02,  5.860754093112697e-01]])

    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T) = \
        finite_difference.compute_VG_APT(0.0001, 0.0001)

    VG_APTs = apyib.vg_apts_fd.VG_APT(parameters, wfn, C, basis, T_list,
        nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
        0.0001, 0.0001)
    vg_apt = np.zeros((3 * H.molecule.natom(), 3))
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            vg_apt[lambd_alpha][beta] = VG_APTs.compute_spatial_vg_apts(lambd_alpha, beta)

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-7)


def test_mp2_vg_apt_fc():
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 6.771669804541723e+00, -6.753070188440530e-11,  7.522715652187378e-13],
     [-3.987272209917419e-11,  7.418859407507148e+00,  2.923371812528155e-12],
     [-3.940919470610871e-14,  8.109853612777127e-15,  8.303606579780928e+00],
     [ 5.130698215567373e-01, -4.133965973873737e-01,  5.585169091165461e-13],
     [-4.134483237775963e-01,  5.122797256895246e-01,  6.465124847444981e-13],
     [-9.058512651619130e-14, -3.241666748913428e-14,  7.898985163200063e-01],
     [ 5.130698215964988e-01,  4.133965974039860e-01, -7.035295566966683e-13],
     [ 4.134483238088709e-01,  5.122797256168375e-01, -3.787944524266859e-13],
     [ 1.114509046495648e-13,  4.981539233967993e-14,  7.898985163200667e-01]])

    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T) = \
        finite_difference.compute_VG_APT(0.0001, 0.0001)

    VG_APTs = apyib.vg_apts_fd.VG_APT(parameters, wfn, C, basis, T_list,
        nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
        0.0001, 0.0001)
    vg_apt = np.zeros((3 * H.molecule.natom(), 3))
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            vg_apt[lambd_alpha][beta] = VG_APTs.compute_spatial_vg_apts(lambd_alpha, beta)

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-7)


@pytest.mark.skip(reason="Too slow.")
def test_cid_vg_apt():
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T) = \
        finite_difference.compute_VG_APT(0.0001, 0.0001)

    VG_APTs = apyib.vg_apts_fd.VG_APT(parameters, wfn, C, basis, T_list,
        nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
        0.0001, 0.0001)
    vg_apt = np.zeros((3 * H.molecule.natom(), 3))
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            vg_apt[lambd_alpha][beta] = VG_APTs.compute_spatial_vg_apts(lambd_alpha, beta, normalization='intermediate')

    assert(np.max(np.abs(vg_apt)) < 1e-7)  # placeholder; fill ref when unblocked


def test_cid_vg_apt_fc():
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 6.805067588913636e+00, -6.783118211320922e-11,  7.246040689384215e-13],
     [-3.893504737859608e-11,  7.436835032811335e+00,  2.921927037703555e-12],
     [-3.905735121463197e-14, -2.421845893061750e-15,  8.311651513966350e+00],
     [ 5.144609701724825e-01, -4.153865960009880e-01,  5.514767579549504e-13],
     [-4.136149782520870e-01,  5.135722891638581e-01,  6.780252659183188e-13],
     [-7.984702296459021e-14, -2.109340349398002e-14,  7.859540474187867e-01],
     [ 5.144609702119652e-01,  4.153865960179714e-01, -7.119294664662436e-13],
     [ 4.136149782828981e-01,  5.135722890917258e-01, -3.839623273153574e-13],
     [ 8.599045866256400e-14,  1.198794304776411e-14,  7.859540474188269e-01]])

    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T) = \
        finite_difference.compute_VG_APT(0.0001, 0.0001)

    VG_APTs = apyib.vg_apts_fd.VG_APT(parameters, wfn, C, basis, T_list,
        nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
        0.0001, 0.0001)
    vg_apt = np.zeros((3 * H.molecule.natom(), 3))
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            vg_apt[lambd_alpha][beta] = VG_APTs.compute_spatial_vg_apts(lambd_alpha, beta)

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-7)
