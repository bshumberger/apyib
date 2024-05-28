import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_rhf_aat():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-6G',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Hessian from CFOUR HF/STO-3G optimized geometry.
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
    AATs = apyib.aats.AAT(parameters, wfn.nbf, wfn.ndocc, C, basis, T_list[2], nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T[2], nuc_neg_T[2], mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T[2], mag_neg_T[2], 0.0001, 0.0001)
    aat = np.zeros((3 * H.molecule.natom(), 3), dtype=np.cdouble)
    for lambd_alpha in range(3 * H.molecule.natom()):
        for beta in range(3):
            aat[lambd_alpha][beta] = AATs.compute_hf_aat(lambd_alpha, beta)

    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)

