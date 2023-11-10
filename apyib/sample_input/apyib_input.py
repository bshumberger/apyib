import psi4
import numpy as np
import scipy.linalg as la
from apyib.utils import run_psi4
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.finite_difference import finite_difference
from apyib.rotational_strength import AAT

# Input geometry.
H2O = """ 
    O   0.000000000000  -0.143225816552   0.000000000000
    H   1.638036840407   1.136548822547  -0.000000000000
    H  -1.638036840407   1.136548822547  -0.000000000000
    units bohr
    symmetry c1
    """

# Set parameters for the calculation.
parameters = {'geom': H2O,
            'basis': 'sto-3g',
            'e_convergence': 1e-13,
            'd_convergence': 1e-13,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 100}

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

print(H.S)

# Set the Hamiltonian defining this instance of the wavefunction object.
wfn = hf_wfn(H)

# Perform the SCF procedure and compute the energy.
E_SCF, E_tot, C = wfn.solve_SCF(parameters)

# Print energies and energy difference between apyib code and Psi4.
print("Electronic Energy: ", E_SCF)
print("Total Energy: ", E_tot)
print("Psi4 Energy: ", p4_rhf_e)
print("Energy Difference between Homemade RHF Code and Psi4: ", E_tot - p4_rhf_e)

#____________________________________________________________________
# => Compute RHF Energies and Wavefunctions for Perturbed Molecule <=
#____________________________________________________________________

print(
    """ 
________________________________________________________________
=> Computing Energies and Wavefunctions for Finite Difference <=
________________________________________________________________
    """
     )

# Compute energies and wavefunctions for finite difference calculations with respect to nuclear displacements.
fin_diff = finite_difference(parameters)
nuc_pos_e, nuc_neg_e, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis = fin_diff.nuclear_displacements(0.0025)

# Compute energies and wavefunctions for for finite difference calculations with respect electric field perturbations.
#fin_diff = finite_difference(parameters)
#elec_pos_e, elec_neg_e, elec_pos_wfns, elec_neg_wfns elec_pos_basis, elec_neg_basis = fin_diff.electric_field_perturbations(0.01)

# Compute energies and wavefunctions for for finite difference calculations with respect magnetic field perturbations.
fin_diff = finite_difference(parameters)
mag_pos_e, mag_neg_e, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis = fin_diff.magnetic_field_perturbations(0.01)

#_____________________________________
# => Compute Finite Difference AATs <=
#_____________________________________

# Compute AO overlap integrals.
AATs = AAT(nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis)
AATs.compute_ao_overlap()

#AATs = AAT(nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis)
#AATs.compute_ao_overlap(molecule)
























