import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_rhf_aug_cc_pvdz_vcd_with_LGOI():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O2_opt_hfapvdz"],
                  'basis': 'aug-cc-pVDZ',
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF aug-cc-pVDZ reference values from collaboration with Jim at Gaussian.
    w_ref = np.array([423.60, 1139.88, 1491.09, 1608.11, 4139.34, 4139.72])
    D_rr_ref = np.array([193.962, 0.825, 105.525, 0.394, 94.571, 27.915])
    D_pp_ref = np.array([96.295, 0.075, 39.236, 0.406, 32.722, 5.688])
    R_rl_ref = np.array([173.595, -2.481, 20.645, -14.220, -38.579, 21.424])
    R_pl_ref = np.array([122.315, -0.747, 13.456, -14.424, -19.746, 9.671])
    R_rl_LGOI_ref = np.array([173.595, -2.481, 22.067, -14.220, -33.569, 21.424])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)
    
    # Compute analytic Hessian.
    analytic_derivative = apyib.analytic_hessian.analytic_derivative(parameters)
    Hessian = analytic_derivative.compute_RHF_Hessian(orbitals = 'non-canonical')
    
    # Compute analytic APT in the length gauge.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    P_LG = analytic_derivative.compute_RHF_APTs_LG(orbitals= 'non-canonical')

    # Compute analytic APT in the velocity gauge.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    P_VG = analytic_derivative.compute_RHF_APTs_VG(orbitals= 'non-canonical')

    # Compute the analytic AAT.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    I = analytic_derivative.compute_RHF_AATs(orbitals='non-canonical')

    # Compute VCD data from length gauge.
    vcd = apyib.vcd.vcd(parameters)
    w, D_rr, D_pp, R_rl, R_pl, R_rl_LGOI = vcd.compute_LGOI_vcd_from_input(Hessian, P_LG, P_VG, I)

    for i in range(len(w)):
        assert(np.abs(w[i] - w_ref[::-1][i]) < 1e-1)
        assert(np.abs(D_rr[i] - D_rr_ref[::-1][i]) < 1e-1)
        assert(np.abs(D_pp[i] - D_pp_ref[::-1][i]) < 1e-1)
        assert(np.abs(R_rl[i] - R_rl_ref[::-1][i]) < 1e-1)
        assert(np.abs(R_pl[i] - R_pl_ref[::-1][i]) < 1e-1)
        assert(np.abs(R_rl_LGOI[i] - R_rl_LGOI_ref[::-1][i]) < 1e-1)

def test_rhf_aug_cc_pvdz_vcd():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O2_opt_hfapvdz"],
                  'basis': 'aug-cc-pVDZ',
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF aug-cc-pVDZ reference values from collaboration with Jim at Gaussian.
    w_ref = np.array([423.60, 1139.88, 1491.09, 1608.11, 4139.34, 4139.72])
    D_rr_ref = np.array([193.962, 0.825, 105.525, 0.394, 94.571, 27.915])
    R_rl_ref = np.array([173.595, -2.481, 20.645, -14.220, -38.579, 21.424])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)
        
    # Compute analytic Hessian.
    analytic_derivative = apyib.analytic_hessian.analytic_derivative(parameters)
    Hessian = analytic_derivative.compute_RHF_Hessian(orbitals = 'non-canonical')
        
    # Compute analytic APT in the length gauge.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    P_LG = analytic_derivative.compute_RHF_APTs_LG(orbitals= 'non-canonical')

    # Compute the analytic AAT.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    I = analytic_derivative.compute_RHF_AATs(orbitals='non-canonical')

    # Compute VCD data from length gauge.
    vcd = apyib.vcd.vcd(parameters)
    w, D_rr, R_rl = vcd.compute_vcd_from_input(Hessian, P_LG, I)

    for i in range(len(w)):
        assert(np.abs(w[i] - w_ref[::-1][i]) < 1e-1)
        assert(np.abs(D_rr[i] - D_rr_ref[::-1][i]) < 1e-1)
        assert(np.abs(R_rl[i] - R_rl_ref[::-1][i]) < 1e-1)
