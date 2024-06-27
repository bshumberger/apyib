import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_mp2_h2o_sto_3g():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-3G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference value.
    psi4_MP2 = -74.9912295643122

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'MP2')
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Compute the MP2 energy and wavefunction.
    wfn_MP2 = apyib.mp2_wfn.mp2_wfn(parameters, wfn)
    apyib_E_MP2, t2 = wfn_MP2.solve_MP2()
     
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic MP2 Energy: ", apyib_E_MP2)
    print("Total Energy: ", apyib_E + apyib_E_MP2 + H.E_nuc)
    print("Psi4 Energy: ", p4_E_tot)
    print("Energy Difference between Homemade MP2 Code and Psi4: ", apyib_E + apyib_E_MP2 + H.E_nuc - p4_E_tot)

    assert(abs(apyib_E + apyib_E_MP2 + H.E_nuc - p4_E_tot) < 1e-11)
    assert(abs(apyib_E + apyib_E_MP2 + H.E_nuc - psi4_MP2) < 1e-11)

def test_mp2_h2o_6_31gd():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G(d)',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference value.
    psi4_MP2 = -76.17533681889488

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'MP2')
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
        
    # Compute the MP2 energy and wavefunction.
    wfn_MP2 = apyib.mp2_wfn.mp2_wfn(parameters, wfn)
    apyib_E_MP2, t2 = wfn_MP2.solve_MP2()
         
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic MP2 Energy: ", apyib_E_MP2)
    print("Total Energy: ", apyib_E + apyib_E_MP2 + H.E_nuc)
    print("Psi4 Energy: ", p4_E_tot)
    print("Energy Difference between Homemade MP2 Code and Psi4: ", apyib_E + apyib_E_MP2 + H.E_nuc - p4_E_tot)

    assert(abs(apyib_E + apyib_E_MP2 + H.E_nuc - p4_E_tot) < 1e-11)
    assert(abs(apyib_E + apyib_E_MP2 + H.E_nuc - psi4_MP2) < 1e-11)

def test_mp2_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G(d)',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference value.
    psi4_MP2 = -76.17533681889488

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]
    
    # Print energies and energy difference between apyib code and Psi4.
    print("Energy Difference between Homemade MP2 Code and Psi4: ", apyib_E_tot - psi4_MP2)

    assert(abs(apyib_E_tot - psi4_MP2) < 1e-11)
