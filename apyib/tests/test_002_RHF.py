import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

#def test_rhf_h2o_sto_3g():
#    # Set parameters for the calculation.
#    parameters = {'geom': moldict["H2O"],
#                  'basis': 'STO-3G',
#                  'method': 'RHF',
#                  'e_convergence': 1e-12,
#                  'd_convergence': 1e-12,
#                  'DIIS': True,
#                  'F_el': [0.0, 0.0, 0.0],
#                  'F_mag': [0.0, 0.0, 0.0],
#                  'max_iterations': 120}
#
#    # Setting RHF reference value.
#    psi4_RHF = -74.9420799281922
#
#    # Run Psi4.
#    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
#    psi4.core.clean()
#
#    # Run apyib.
#    H = apyib.hamiltonian.Hamiltonian(parameters) 
#    wfn = apyib.hf_wfn.hf_wfn(H)
#    apyib_E, apyib_E_tot, apyib_wfn = wfn.solve_SCF(parameters)
#    
#    # Print energies and energy difference between apyib code and Psi4.
#    print("apyib Electronic Hartree-Fock Energy: ", apyib_E)
#    print("apyib Total Energy: ", apyib_E_tot)
#    print("Psi4 Total Energy: ", p4_E_tot)
#    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E_tot - p4_E_tot)
#
#    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
#    assert(abs(apyib_E_tot - psi4_RHF) < 1e-11)
#
#def test_rhf_h2o_6_31gd():
#    # Set parameters for the calculation.
#    parameters = {'geom': moldict["H2O"],
#                  'basis': '6-31G(d)',
#                  'method': 'RHF',
#                  'e_convergence': 1e-12,
#                  'd_convergence': 1e-12,
#                  'DIIS': True,
#                  'F_el': [0.0, 0.0, 0.0],
#                  'F_mag': [0.0, 0.0, 0.0],
#                  'max_iterations': 120}
#
#    # Setting RHF reference value.
#    psi4_RHF = -75.97474825544526
#    
#    # Run Psi4.
#    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
#    psi4.core.clean()
#
#    # Run apyib.
#    H = apyib.hamiltonian.Hamiltonian(parameters) 
#    wfn = apyib.hf_wfn.hf_wfn(H)
#    apyib_E, apyib_E_tot, apyib_wfn = wfn.solve_SCF(parameters)
#    
#    # Print energies and energy difference between apyib code and Psi4.
#    print("apyib Electronic Hartree-Fock Energy: ", apyib_E)
#    print("apyib Total Energy: ", apyib_E_tot)
#    print("Psi4 Total Energy: ", p4_E_tot)
#    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E_tot - p4_E_tot)
#
#    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
#    assert(abs(apyib_E_tot - psi4_RHF) < 1e-11)

def test_rhf_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G(d)',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference value.
    psi4_RHF = -75.97474825544526

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]
    
    # Print energies and energy difference between apyib code and Psi4.    
    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E_tot - psi4_RHF)

    assert(abs(apyib_E_tot - psi4_RHF) < 1e-11)

