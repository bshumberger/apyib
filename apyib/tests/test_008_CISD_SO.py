import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_cisd_h2o_6_31g():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    psi4_CISD = -76.09503651362023

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CISD')
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Compute the MP2 energy and wavefunction.
    wfn_CI = apyib.ci_wfn.ci_wfn(parameters, wfn)
    apyib_E_CISD, t1, t2 = wfn_CI.solve_CISD_SO()
    
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic CISD Energy: ", apyib_E_CISD)
    print("Total Energy: ", apyib_E + H.E_nuc + apyib_E_CISD)
    print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E + H.E_nuc + apyib_E_CISD - p4_E_tot)

    assert(abs(apyib_E + H.E_nuc + apyib_E_CISD - p4_E_tot) < 1e-11)
    assert(abs(apyib_E + H.E_nuc + apyib_E_CISD - psi4_CISD) < 1e-11)

#@pytest.mark.skip(reason="Too slow.")
def test_cisd_h2o_cc_pvdz():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    psi4_CISD = -76.20375874728018

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CISD')
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Compute the MP2 energy and wavefunction.
    wfn_CI = apyib.ci_wfn.ci_wfn(parameters, wfn)
    apyib_E_CISD, t1, t2 = wfn_CI.solve_CISD_SO()
    
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic CISD Energy: ", apyib_E_CISD)
    print("Total Energy: ", apyib_E + H.E_nuc + apyib_E_CISD)
    print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E + H.E_nuc + apyib_E_CISD - p4_E_tot)

    assert(abs(apyib_E + H.E_nuc + apyib_E_CISD - p4_E_tot) < 1e-11)
    assert(abs(apyib_E + H.E_nuc + apyib_E_CISD - psi4_CISD) < 1e-11)

def test_cisd_SO_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CID reference value.
    psi4_CISD = -76.09503651362023

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E_tot - psi4_CISD)

    assert(abs(apyib_E_tot - psi4_CISD) < 1e-11)
