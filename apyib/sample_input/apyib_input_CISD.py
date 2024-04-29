import psi4
import numpy as np
import scipy.linalg as la
from apyib.utils import run_psi4
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn
from apyib.finite_difference import finite_difference
from apyib.rotational_strength import AAT
from apyib.rotational_strength import perm_parity
from apyib.rotational_strength import compute_mo_overlap
from apyib.rotational_strength import compute_phase

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
            'basis': '6-31G',
            'method': 'CISD',
            'e_convergence': 1e-12,
            'd_convergence': 1e-12,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 120}

# Run Psi4.
p4_rhf_e, p4_rhf_wfn = run_psi4(parameters)

#___________________________________________________
# => Compute Hartree-Fock Energy and Wavefunction <=
#___________________________________________________

print(
    """ 
___________________________________
=> Computing Hartree-Fock Energy <=
___________________________________
    """
     )

# Build the Hamiltonian in the AO basis.
H = Hamiltonian(parameters)

# Set the Hamiltonian defining this instance of the wavefunction object.
wfn = hf_wfn(H)

# Perform the SCF procedure and compute the energy.
E_SCF, E_tot, C = wfn.solve_SCF(parameters)

# Store unperturbed basis set and wavefunction.
unperturbed_basis = H.basis_set
unperturbed_wfn = C

# Print energies and energy difference between apyib code and Psi4.
print("Electronic Hartree-Fock Energy: ", E_SCF)
print("Total Energy: ", E_tot)
print("Psi4 Energy: ", p4_rhf_e)
print("Energy Difference between Homemade RHF Code and Psi4: ", E_tot - p4_rhf_e)


#_____________________________________________________________________
# => Compute CISD Energy and Wavefunction in the Spin Orbital Basis <=
#_____________________________________________________________________

print(
    """ 
____________________________________________________
=> Computing CISD Energy in the Spin Orbital Basis <=
____________________________________________________
    """
     )   

# Compute the MP2 energy and wavefunction in the spin orbital basis.
wfn_CI_SO = ci_wfn(parameters, E_SCF, E_tot, C)
E_CISD_SO, t1_SO, t2_SO = wfn_CI_SO.solve_CISD_SO()

# Run Psi4 MP2 for comparison.
p4_cisd_e, p4_cisd_wfn = run_psi4(parameters, 'CISD')

# Print energies and energy difference between apyib code and Psi4.
print("Electronic Hartree-Fock Energy: ", E_SCF)
print("Electronic CISD SO Energy: ", E_CISD_SO)
print("Total SO Energy: ", E_tot + E_CISD_SO)
print("Psi4 Energy: ", p4_cisd_e)
print("Energy Difference between Homemade CISD Code and Psi4: ", E_tot + E_CISD_SO - p4_cisd_e)




