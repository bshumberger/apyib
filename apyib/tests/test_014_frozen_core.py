import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

@pytest.mark.skip(reason="Not ready yet.")
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

    # Setting CID reference value.
    psi4_CISD = -76.20195791202764

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
    print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E_tot - psi4_CISD)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_CISD) < 1e-11)

