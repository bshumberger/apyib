import psi4
import numpy as np
import scipy.linalg as la
from apyib.utils import run_psi4
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.finite_difference import finite_difference
from apyib.rotational_strength import AAT
from apyib.rotational_strength import perm_parity
#from apyib.rotational_strength import heaperm
from apyib.rotational_strength import compute_mo_overlap
#from apyib.rotational_strength import compute_hf_overlap
from apyib.rotational_strength import compute_phase

# Input geometry.
H2O2 = """
    O       0.0000000000        1.3192641900       -0.0952542913
    O      -0.0000000000       -1.3192641900       -0.0952542913
    H       1.6464858700        1.6841036400        0.7620343300
    H      -1.6464858700       -1.6841036400        0.7620343300
    symmetry c1
    units bohr
    noreorient
    no_com
    """

ethylene_oxide = """
    O       0.0000000000        0.0000000000        1.6119363900
    C       0.0000000000       -1.3813890400       -0.7062143040
    C       0.0000000000        1.3813905700       -0.7062514120
    H      -1.7489765900       -2.3794725300       -1.1019539000
    H       1.7489765900       -2.3794725300       -1.1019539000
    H       1.7489765900        2.3794634300       -1.1020178200
    H      -1.7489765900        2.3794634300       -1.1020178200
    symmetry c1
    no_reorient
    no_com
    units bohr
    """

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
parameters = {'geom': H2O2,
            'basis': 'STO-6G',
            'e_convergence': 1e-13,
            'd_convergence': 1e-13,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 100}

# Run Psi4.
p4_rhf_e, p4_rhf_wfn = run_psi4(parameters)

# Print dipole to Psi4 output.
psi4.oeprop(p4_rhf_wfn, 'DIPOLE')

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

# Setting up perturbations/displacements.
nuclear_displacement = 0.001
magnetic_field_perturbation = 0.001

# Initializing finite difference object.
fin_diff = finite_difference(parameters)

# Compute energies and wavefunctions for finite difference calculations with respect to nuclear displacements.
nuc_pos_e, nuc_neg_e, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis = fin_diff.nuclear_displacements(nuclear_displacement)

# Compute energies and wavefunctions for for finite difference calculations with respect electric field perturbations.
#elec_pos_e, elec_neg_e, elec_pos_wfns, elec_neg_wfns, elec_pos_basis, elec_neg_basis = fin_diff.electric_field_perturbations(0.001)

# Compute energies and wavefunctions for for finite difference calculations with respect magnetic field perturbations.
mag_pos_e, mag_neg_e, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis = fin_diff.magnetic_field_perturbations(magnetic_field_perturbation)

#_____________________________________
# => Compute Finite Difference AATs <=
#_____________________________________

# Compute AO overlap integrals.
AATs = AAT(wfn.nbf, wfn.ndocc, unperturbed_wfn, unperturbed_basis, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis, nuclear_displacement, magnetic_field_perturbation)


#print("Electronic Contribution to the AATS:")
#
#for alpha in range(len(nuc_pos_wfns)):
#    I_x = AATs.compute_aat(alpha, 0)
#    I_y = AATs.compute_aat(alpha, 1)
#    I_z = AATs.compute_aat(alpha, 2)
#    print("{0:8.7f}, {1:8.7f}, {2:8.7f}" .format(I_x.imag, I_y.imag, I_z.imag))

print("Electronic Contribution to the AATS:")
for alpha in range(len(nuc_pos_wfns)):
    I_x1 = AATs.compute_aat1(alpha, 0)
    I_y1 = AATs.compute_aat1(alpha, 1)
    I_z1 = AATs.compute_aat1(alpha, 2)
    print("{0:8.7f}, {1:8.7f}, {2:8.7f}" .format(I_x1.imag, I_y1.imag, I_z1.imag))

