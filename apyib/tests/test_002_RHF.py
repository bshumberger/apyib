import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_rhf_h2o_sto_3g():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': 'STO-3G',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference value.
    psi4_RHF = -74.9420799281922

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Print energies and energy difference between apyib code and Psi4.
    print("apyib Electronic Hartree-Fock Energy: ", apyib_E)
    print("apyib Total Energy: ", apyib_E + H.E_nuc)
    print("Psi4 Total Energy: ", p4_E_tot)
    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E + H.E_nuc - p4_E_tot)

    assert(abs(apyib_E + H.E_nuc - p4_E_tot) < 1e-11)
    assert(abs(apyib_E + H.E_nuc - psi4_RHF) < 1e-11)

def test_rhf_h2o_6_31gd():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G(d)',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'freeze_core': False,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference value.
    psi4_RHF = -75.97474825544526
    
    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
    psi4.core.clean()

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Print energies and energy difference between apyib code and Psi4.
    print("apyib Electronic Hartree-Fock Energy: ", apyib_E)
    print("apyib Total Energy: ", apyib_E + H.E_nuc)
    print("Psi4 Total Energy: ", p4_E_tot)
    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E + H.E_nuc - p4_E_tot)

    assert(abs(apyib_E + H.E_nuc- p4_E_tot) < 1e-11)
    assert(abs(apyib_E + H.E_nuc- psi4_RHF) < 1e-11)

def test_rhf_energy():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G(d)',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
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

def test_rhf_energy_methyloxirane_6_31G():
    # Set parameters for the calculation.
    S_methyloxirane_aug_cc_pVDZ = """
    C  -0.406643263371948   0.103169698978202  -0.932904009322372
    O   1.550888890337005  -1.412171718063139   0.260072952566956
    C   1.810064371203058   1.327408273345544   0.226984115881429
    H   3.344838669765579   2.027224623297944  -0.964879943170253
    H   1.599930911340288   2.226009101570543   2.076678955772736
    C  -2.946240158335548   0.082079933386530   0.348469914386564
    H  -4.122138549897469   1.649720715533630  -0.352252535246005
    H  -3.940556043431783  -1.702386417970233  -0.037902074947365
    H  -2.712874671266827   0.277413155894713   2.403143387960141
    H  -0.412850332112492  -0.076749763394509  -2.996238617375079

    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': S_methyloxirane_aug_cc_pVDZ,
                  'basis': '6-31G',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)
    psi4.core.clean()

    # Setting RHF reference value.
    psi4_RHF = p4_E_tot

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]
    
    # Print energies and energy difference between apyib code and Psi4.    
    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E_tot - psi4_RHF)

    assert(abs(apyib_E_tot - psi4_RHF) < 1e-11)
