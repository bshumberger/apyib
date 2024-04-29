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

disp_H2O = """
    O   1.000000000000   0.856774183448   1.000000000000
    H   2.638036840407   2.136548822547   1.000000000000
    H  -0.638036840407   2.136548822547   1.000000000000
    units bohr
    symmetry c1
    no_reorient
    nocom
    """

H4 = """
    H
    H 1 0.75
    H 2 1.5 1 90.0
    H 3 0.75 2 90.0 1 60.0
    symmetry c1
    no_reorient
    nocom
    """

H2_2 = """
    H -0.708647297046685  1.417294594093371 -0.613706561565607
    H  0.                 1.417294594093371  0.613706561565607
    H  0.                -1.417294594093371  0.613706561565607
    H  0.708647297046686 -1.417294594093371 -0.613706561565607
    no_com
    no_reorient
    symmetry c1
    units bohr
    """



# Set parameters for the calculation.
parameters = {'geom': H2O,
            'basis': 'STO-6G',
            'method': 'RHF',
            'e_convergence': 1e-12,
            'd_convergence': 1e-12,
            'DIIS': True,
            'F_el': [0.0, 0.0, 0.0],
            'F_mag': [0.0, 0.0, 0.0],
            'max_iterations': 120}

# Run Psi4.
p4_rhf_e, p4_rhf_wfn = run_psi4(parameters)

# Print dipole to Psi4 output.
#psi4.oeprop(p4_rhf_wfn, 'DIPOLE')

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

#____________________________________________________________________________________________
# => Compute Second-Order Moller-Plesset Perturbation Theory (MP2) Energy and Wavefunction <=
#____________________________________________________________________________________________

print(
    """ 
____________________________________________________________________________
=> Computing Second-Order Moller-Plesset Perturbation Theory (MP2) Energy <=
____________________________________________________________________________
    """
     )

# Compute the MP2 energy and wavefunction.
wfn_MP2 = mp2_wfn(parameters, E_SCF, E_tot, C)
E_MP2, t2mp = wfn_MP2.solve_MP2()

# Run Psi4 MP2 for comparison.
p4_mp2_e, p4_mp2_wfn = run_psi4(parameters, 'MP2')

# Print energies and energy difference between apyib code and Psi4.
print("Electronic Hartree-Fock Energy: ", E_SCF)
print("Electronic MP2 Energy: ", E_MP2)
print("Total Energy: ", E_tot + E_MP2)
print("Psi4 Energy: ", p4_mp2_e)
print("Energy Difference between Homemade MP2 Code and Psi4: ", E_tot + E_MP2 - p4_mp2_e)

#____________________________________________________________________
# => Compute MP2 Energy and Wavefunction in the Spin Orbital Basis <=
#____________________________________________________________________

print(
    """ 
____________________________________________________
=> Computing MP2 Energy in the Spin Orbital Basis <=
____________________________________________________
    """
     )

# Compute the MP2 energy and wavefunction in the spin orbital basis.
wfn_MP2_SO = mp2_wfn(parameters, E_SCF, E_tot, C)
E_MP2_SO, t2mp_SO = wfn_MP2_SO.solve_MP2_SO()

# Print energies and energy difference between apyib code and Psi4.
print("Electronic Hartree-Fock Energy: ", E_SCF)
print("Electronic MP2 SO Energy: ", E_MP2_SO)
print("Total SO Energy: ", E_tot + E_MP2_SO)
print("Psi4 Energy: ", p4_mp2_e)
print("Energy Difference between Homemade MP2 Code and Psi4: ", E_tot + E_MP2_SO - p4_mp2_e)
print("Energy Difference Between MP2 and Spin-Obital MP2: ", E_MP2 - E_MP2_SO)


#________________________________________________________________________
# => Compute Configuration Interaction Doubles Energy and Wavefunction <=
#________________________________________________________________________

print(
    """ 
______________________________________________________________
=> Computing Configuration Interaction Doubles (CID) Energy <=
______________________________________________________________
    """
     )   

# Compute the CID energy and T2 amplitudes.
wfn_CI = ci_wfn(parameters, E_SCF, E_tot, C)
E_CID, t2 = wfn_CI.solve_CID()

# Print energies for apyib code.
print("Electronic Hartree-Fock Energy: ", wfn_CI.E_SCF)
print("Electronic CID Energy: ", E_CID)
print("Total Energy: ", wfn_CI.E_tot + E_CID)

#____________________________________________________________________
# => Compute CID Energy and Wavefunction in the Spin Orbital Basis <=
#____________________________________________________________________

print(
    """ 
____________________________________________________
=> Computing CID Energy in the Spin Orbital Basis <=
____________________________________________________
    """
     )   

# Compute the MP2 energy and wavefunction in the spin orbital basis.
wfn_CI_SO = ci_wfn(parameters, E_SCF, E_tot, C)
E_CID_SO, t2_SO = wfn_CI_SO.solve_CID_SO()

# Print energies and energy difference between apyib code and Psi4.
print("Electronic Hartree-Fock Energy: ", E_SCF)
print("Electronic CID SO Energy: ", E_CID_SO)
print("Total SO Energy: ", E_tot + E_CID_SO)
print("Energy Difference Between CID and Spin-Obital CID: ", E_CID - E_CID_SO)

##_____________________________________________________________________
## => Compute CISD Energy and Wavefunction in the Spin Orbital Basis <=
##_____________________________________________________________________
#
#print(
#    """ 
#____________________________________________________
#=> Computing CISD Energy in the Spin Orbital Basis <=
#____________________________________________________
#    """
#     )   
#
## Compute the MP2 energy and wavefunction in the spin orbital basis.
#wfn_CI_SO = ci_wfn(parameters, E_SCF, E_tot, C)
#E_CISD_SO, t1_SO, t2_SO = wfn_CI_SO.solve_CISD_SO()
#
## Run Psi4 MP2 for comparison.
#p4_cisd_e, p4_cisd_wfn = run_psi4(parameters, 'CISD')
#
## Print energies and energy difference between apyib code and Psi4.
#print("Electronic Hartree-Fock Energy: ", E_SCF)
#print("Electronic CISD SO Energy: ", E_CISD_SO)
#print("Total SO Energy: ", E_tot + E_CISD_SO)
#print("Psi4 Energy: ", p4_cisd_e)
#print("Energy Difference between Homemade MP2 Code and Psi4: ", E_tot + E_CISD_SO - p4_cisd_e)


#____________________________________________________________________
# => Compute Energies and Wavefunctions for Perturbed Molecule <=
#____________________________________________________________________

print(
    """ 
________________________________________________________________
=> Computing Energies and Wavefunctions for Finite Difference <=
________________________________________________________________
    """
     )

# Setting up perturbations/displacements.
nuclear_displacement = 0.0001
electric_field_perturbation = 0.0001
magnetic_field_perturbation = 0.0001

# Initializing finite difference object.
#fin_diff_SO = finite_difference(parameters, unperturbed_basis, unperturbed_wfn)

# Compute energies and wavefunctions for finite difference calculations with respect to nuclear displacements.
#nuc_pos_e_SO, nuc_neg_e_SO, nuc_pos_wfns_SO, nuc_neg_wfns_SO, nuc_pos_basis_SO, nuc_neg_basis_SO, nuc_pos_t2_SO, nuc_neg_t2_SO = fin_diff_SO.nuclear_displacements_SO(nuclear_displacement)
# Compute energies and wavefunctions for for finite difference calculations with respect electric field perturbations.
#elec_pos_e_SO, elec_neg_e_SO, elec_pos_wfns_SO, elec_neg_wfns_SO, elec_pos_basis_SO, elec_neg_basis_SO, elec_pos_t2_SO, elec_neg_t2_SO = fin_diff_SO.electric_field_perturbations_SO(electric_field_perturbation)
# Compute energies and wavefunctions for for finite difference calculations with respect magnetic field perturbations.
#mag_pos_e_SO, mag_neg_e_SO, mag_pos_wfns_SO, mag_neg_wfns_SO, mag_pos_basis_SO, mag_neg_basis_SO, mag_pos_t2_SO, mag_neg_t2_SO = fin_diff_SO.magnetic_field_perturbations_SO(magnetic_field_perturbation)

# Initializing finite difference object.
fin_diff = finite_difference(parameters, unperturbed_basis, unperturbed_wfn)

# Compute energies and wavefunctions for finite difference calculations with respect to nuclear displacements.
nuc_pos_e, nuc_neg_e, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, nuc_pos_t2, nuc_neg_t2 = fin_diff.nuclear_displacements(nuclear_displacement)
# Compute energies and wavefunctions for for finite difference calculations with respect electric field perturbations.
#elec_pos_e, elec_neg_e, elec_pos_wfns, elec_neg_wfns, elec_pos_basis, elec_neg_basis, elec_pos_t2, elec_neg_t2 = fin_diff.electric_field_perturbations(electric_field_perturbation)
# Compute energies and wavefunctions for for finite difference calculations with respect magnetic field perturbations.
mag_pos_e, mag_neg_e, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis, mag_pos_t2, mag_neg_t2 = fin_diff.magnetic_field_perturbations(magnetic_field_perturbation)


#_____________________________________
# => Compute Finite Difference AATs <=
#_____________________________________

print(
    """ 
_____________________________________________________________
=> Computing Finite Difference Atomic Axial Tensors (AATs) <=
_____________________________________________________________
    """
     )

## Storing unperturbed T2 amplitudes.
#if parameters['method'] == 'RHF':
#    unperturbed_t2_SO = np.zeros((2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
#elif parameters['method'] == 'MP2':
#    unperturbed_t2_SO = t2mp_SO
#elif parameters['method'] == 'CID':
#    unperturbed_t2_SO = t2_SO

# Storing unperturbed T2 amplitudes.
if parameters['method'] == 'RHF':
    unperturbed_t2 = np.zeros((wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
elif parameters['method'] == 'MP2':
    unperturbed_t2 = t2mp
elif parameters['method'] == 'CID':
    unperturbed_t2 = t2


## Compute AO overlap integrals.
#AATs_SO = AAT(parameters, wfn.nbf, wfn.ndocc, unperturbed_wfn, unperturbed_basis, unperturbed_t2_SO, nuc_pos_wfns_SO, nuc_neg_wfns_SO, nuc_pos_basis_SO, nuc_neg_basis_SO, nuc_pos_t2_SO, nuc_neg_t2_SO, mag_pos_wfns_SO, mag_neg_wfns_SO, mag_pos_basis_SO, mag_neg_basis_SO, mag_pos_t2_SO, mag_neg_t2_SO, nuclear_displacement, magnetic_field_perturbation)

# Compute AO overlap integrals.
AATs = AAT(parameters, wfn.nbf, wfn.ndocc, unperturbed_wfn, unperturbed_basis, unperturbed_t2, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, nuc_pos_t2, nuc_neg_t2, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis, mag_pos_t2, mag_neg_t2, nuclear_displacement, magnetic_field_perturbation)

print("Hartree-Fock AATs")
print("Electronic Contribution to the AATs:")
for alpha in range(len(nuc_pos_wfns)):
    I_x1 = AATs.compute_hf_aat(alpha, 0)
    I_y1 = AATs.compute_hf_aat(alpha, 1)
    I_z1 = AATs.compute_hf_aat(alpha, 2)
    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x1.imag, I_y1.imag, I_z1.imag))

print("\nCorrelated AATs - Spin Adapted Shumberger Approach")
print("Electronic contribution to AAT (a.u.):\n")
for alpha in range(len(nuc_pos_wfns)):
    I_x4 = AATs.compute_cid_aat(alpha, 0)
    I_y4 = AATs.compute_cid_aat(alpha, 1)
    I_z4 = AATs.compute_cid_aat(alpha, 2)
    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x4.imag, I_y4.imag, I_z4.imag))

#print("\nCorrelated AATs - Shumberger Approach")
#print("Electronic contribution to AAT (a.u.):\n")
#for alpha in range(len(nuc_pos_wfns_SO)):
#    I_x2 = AATs_SO.compute_cid_aat_SO_BS(alpha, 0)
#    I_y2 = AATs_SO.compute_cid_aat_SO_BS(alpha, 1)
#    I_z2 = AATs_SO.compute_cid_aat_SO_BS(alpha, 2)
#    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x2.imag, I_y2.imag, I_z2.imag))

##print("\nCorrelated AATs - Crawford Approach")
##print("Electronic contribution to AAT (a.u.):\n")
##for alpha in range(len(nuc_pos_wfns)):
##    I_x3 = AATs_SO.compute_cid_aat_SO_Crawdad(alpha, 0)
##    I_y3 = AATs_SO.compute_cid_aat_SO_Crawdad(alpha, 1)
##    I_z3 = AATs_SO.compute_cid_aat_SO_Crawdad(alpha, 2)
##    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x3.imag, I_y3.imag, I_z3.imag))
#
#
##psi_h = psi4.constants.h
##psi_hbar = psi_h / (2 * np.pi)
##psi_dipmom_au2si = psi4.constants.dipmom_au2si
##psi_m2angstroms = 10**10
#
##print("Hartree-Fock AATs")
##print("Electronic contribution to AAT (T^-1 A^-1 * 10^-6):\n")
##for alpha in range(len(nuc_pos_wfns)):
##    I_x = AATs.compute_hf_aat(alpha, 0) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    I_y = AATs.compute_hf_aat(alpha, 1) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    I_z = AATs.compute_hf_aat(alpha, 2) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x.imag, I_y.imag, I_z.imag))
#
##print("\nCorrelated AATs from Spin-Orbitals")
##print("Electronic contribution to AAT (T^-1 A^-1 * 10^-6):\n")
##for alpha in range(len(nuc_pos_wfns)):
##    I_x = AATs_SO.compute_cid_aat_SO(alpha, 0) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    I_y = AATs_SO.compute_cid_aat_SO(alpha, 1) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    I_z = AATs_SO.compute_cid_aat_SO(alpha, 2) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x.imag, I_y.imag, I_z.imag))
#
##print("\nCorrelated AATs")
##print("Electronic contribution to AAT (T^-1 A^-1 * 10^-6):\n")
##for alpha in range(len(nuc_pos_wfns)):
##    I_x = AATs.compute_cid_aat(alpha, 0) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    I_y = AATs.compute_cid_aat(alpha, 1) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    I_z = AATs.compute_cid_aat(alpha, 2) * psi_dipmom_au2si * (1 / psi_hbar) * (1 / psi_m2angstroms) * 10**6
##    print("{0:8.12f}, {1:8.12f}, {2:8.12f}" .format(I_x.imag, I_y.imag, I_z.imag))















