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
            'basis': 'cc-pVDZ',
            'method': 'RHF',
            'e_convergence': 1e-12,
            'd_convergence': 1e-12,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 120}

def test_cisd_h2o_so():
    # Setting CID reference value.
    #g09_CISD = -76.09503650364046
    #g09_CISD = -76.09503650363168

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters)

    # Run apyib.
    #H = apyib.hamiltonian.Hamiltonian(parameters)
    #wfn = apyib.hf_wfn.hf_wfn(H)
    #apyib_E, apyib_E_tot, apyib_wfn = wfn.solve_SCF(parameters)
    #
    ## Compute the MP2 energy and wavefunction.
    #wfn_CI = apyib.ci_wfn.ci_wfn(parameters, apyib_E, apyib_E_tot, apyib_wfn)
    #apyib_E_CISD, t2 = wfn_CI.solve_CISD_SO()
    #
    ## Print energies and energy difference between apyib code and Psi4.
    #print("Electronic Hartree-Fock Energy: ", apyib_E)
    #print("Electronic CID Energy: ", apyib_E_CISD)
    #print("Total Energy: ", apyib_E_tot + apyib_E_CISD)
    #print("Energy Difference between Homemade CISD Code and Reference: ", apyib_E_tot + apyib_E_CISD - g09_CISD)

    #assert(abs(apyib_E_tot + apyib_E_CISD - p4_E_tot) < 1e-12)
    #assert(abs(apyib_E_tot + apyib_E_CISD - g09_CISD) < 1e-12)


