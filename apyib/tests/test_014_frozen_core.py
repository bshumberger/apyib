import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_mp2_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference value.
    psi4_MP2 = -76.20202577979143

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'MP2')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic MP2 Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade MP2 Code and Reference: ", apyib_E_tot - psi4_MP2)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_MP2) < 1e-11)

def test_mp2_SO_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'MP2_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference value.
    psi4_MP2 = -76.20202577979143

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'MP2')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic MP2 Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade MP2 Code and Reference: ", apyib_E_tot - psi4_MP2)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_MP2) < 1e-11)

def test_cid_energy():
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

#@pytest.mark.skip(reason="Too long.")
def test_cid_SO_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CID_SO',
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
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CID Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CID Code and Reference: ", apyib_E_tot - g09_CID)

    assert(abs(apyib_E_tot - g09_CID) < 1e-11)

def test_cisd_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CISD',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CISD reference value.
    g09_CISD = -76.20195791202802

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CISD')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters,print_level=1)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CISD Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E_tot - g09_CISD)

    assert(abs(apyib_E_tot - g09_CISD) < 1e-11)

#@pytest.mark.skip(reason="Too long.")
def test_cisd_SO_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'cc-pVDZ',
                  'method': 'CISD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CISD reference value.
    g09_CISD = -76.20195791202802

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CISD')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CISD Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E_tot - g09_CISD)

    assert(abs(apyib_E_tot - g09_CISD) < 1e-11)


