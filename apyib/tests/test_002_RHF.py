import psi4
import numpy as np
import apyib

# Input geometry.
H2O = """ 
    O   0.000000000000  -0.143225816552   0.000000000000
    H   1.638036840407   1.136548822547  -0.000000000000
    H  -1.638036840407   1.136548822547  -0.000000000000
    units bohr
    symmetry c1
    no_reorient
    nocom
    """


# Set parameters for the calculation.
parameters = {'geom': H2O,
            'basis': 'STO-3G',
            'method': 'RHF',
            'e_convergence': 1e-12,
            'd_convergence': 1e-12,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 120}

def test_rhf_h2o():
    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_E_tot, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Print energies and energy difference between apyib code and Psi4.
    print("apyib Electronic Hartree-Fock Energy: ", apyib_E)
    print("apyib Total Energy: ", apyib_E_tot)
    print("Psi4 Total Energy: ", p4_E_tot)
    print("Energy Difference between Homemade RHF Code and Psi4: ", apyib_E_tot - p4_E_tot)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-12)


