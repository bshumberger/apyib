import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_cid_h2o_sto_3g():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-3G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    g09_CID = -75.01073817893661

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Compute the MP2 energy and wavefunction.
    wfn_CI = apyib.ci_wfn.ci_wfn(parameters, wfn)
    apyib_E_CID, t2 = wfn_CI.solve_CID()
    
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic CID Energy: ", apyib_E_CID)
    print("Total Energy: ", apyib_E + apyib_E_CID + H.E_nuc)
    print("Energy Difference between Homemade CID Code and Reference: ", apyib_E + apyib_E_CID + H.E_nuc - g09_CID)

    assert(abs(apyib_E + apyib_E_CID + H.E_nuc - g09_CID) < 1e-11)

def test_cid_h2o_cc_pvdz():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    #g09_CID = -76.17417471109950
    c4scf = -75.98979581991861
    c4ci = -0.21279410950205
    c4_CID = c4scf + c4ci

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
        
    # Compute the MP2 energy and wavefunction.
    wfn_CI = apyib.ci_wfn.ci_wfn(parameters, wfn)
    apyib_E_CID, t2 = wfn_CI.solve_CID()
        
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic CID Energy: ", apyib_E_CID)
    print("Total Energy: ", apyib_E + apyib_E_CID + H.E_nuc)
    print("Energy Difference between Homemade CID Code and Reference: ", apyib_E + apyib_E_CID + H.E_nuc - c4_CID)

    assert(abs(apyib_E + apyib_E_CID + H.E_nuc - c4_CID) < 1e-11)

def test_cid_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    c4scf = -75.98979581991861
    c4ci = -0.21279410950205
    c4_CID = c4scf + c4ci

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Energy Difference between Homemade CID Code and Reference: ", apyib_E_tot - c4_CID)

    assert(abs(apyib_E_tot - c4_CID) < 1e-11)

def test_cid_energy_fc():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    g09_CID = -76.20078548433332

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters,print_level=1)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CID Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CID Code and Reference: ", apyib_E_tot - g09_CID)

    assert(abs(apyib_E_tot - g09_CID) < 1e-11)
