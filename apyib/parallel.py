"""Contains the functions associated with computing the AATs in parallel."""

import psi4
import numpy as np
import math
import itertools as it
import time
import multiprocessing as mp
from apyib.hamiltonian import Hamiltonian
from apyib.energy import energy
from apyib.hf_wfn import hf_wfn
from apyib.fin_diff import finite_difference
from apyib.aats import AAT

def compute_parallel_aats(parameters, nuc_pert_strength, mag_pert_strength, normalization='full'):
    # Compute energy.
    E_list, T_list, C, basis = energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print("Total Energy: ", E_tot)

    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)

    # Compute finite difference AATs inputs.
    aat_finite_difference = finite_difference(parameters, basis, C)
    nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T = aat_finite_difference.compute_AAT(nuc_pert_strength, mag_pert_strength)

    # Compute finite difference AATs.
    AATs = AAT(parameters, wfn, C, basis, T_list, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, nuc_pert_strength, mag_pert_strength)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    pool = mp.Pool()
    lambd_alpha = np.array(range(3 * H.molecule.natom()))
    beta = np.array(range(3))
    lab = []
    for a in lambd_alpha:
        for b in beta:
            lab.append([a, b, normalization]) 
    if parameters['method'] == 'RHF' or parameters['method'] == 'MP2' or parameters['method'] == 'CID' or parameters['method'] == 'CISD':
        I = pool.starmap_async(AATs.compute_spatial_aats, lab,)
    if parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
        I = pool.starmap_async(AATs.compute_SO_aats, lab,)
    pool.close()
    pool.join()
    I = np.reshape(np.array(I.get()), (3*H.molecule.natom(),3))
    print(I, "\n")

    return I
