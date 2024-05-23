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
            'method': 'MP2',
            'e_convergence': 1e-12,
            'd_convergence': 1e-12,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 120}

def test_mp2_h2o_so():
    # Setting MP2 reference value.
    g09_MP2 = -74.99122956431164

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'MP2')

    # Run apyib.
    H = apyib.hamiltonian.Hamiltonian(parameters) 
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_E_tot, apyib_wfn = wfn.solve_SCF(parameters)
    
    # Compute the MP2 energy and wavefunction.
    wfn_MP2 = apyib.mp2_wfn.mp2_wfn(parameters, apyib_E, apyib_E_tot, apyib_wfn)
    apyib_E_MP2, t2 = wfn_MP2.solve_MP2_SO()
     
    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", apyib_E)
    print("Electronic MP2 Energy: ", apyib_E_MP2)
    print("Total Energy: ", apyib_E_tot + apyib_E_MP2)
    print("Psi4 Energy: ", p4_E_tot)
    print("Energy Difference between Homemade MP2 Code and Psi4: ", apyib_E_tot + apyib_E_MP2 - p4_E_tot)

    assert(abs(apyib_E_tot + apyib_E_MP2 - p4_E_tot) < 1e-12)
    assert(abs(apyib_E_tot + apyib_E_MP2 - g09_MP2) < 1e-12)
