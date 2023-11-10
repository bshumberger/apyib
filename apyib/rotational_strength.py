"""Contains the class and functions associated with computing the rotational strength for VCD calculations by finite difference at the Hartree-Fock level of theory."""

import psi4
import numpy as np



class AAT(object):
    """
    The atomic axial tensor object computed by finite difference.
    """
    def __init__(self, nuc_pos_wfn, nuc_neg_wfn, nuc_pos_basis, nuc_neg_basis, mag_pos_wfn, mag_neg_wfn, mag_pos_basis, mag_neg_basis):

        # Basis sets and wavefunctions from calculations with respect to nuclear displacements.
        self.nuc_pos_basis = nuc_pos_basis
        self.nuc_neg_basis = nuc_neg_basis
        self.nuc_pos_wfn = nuc_pos_wfn
        self.nuc_neg_wfn = nuc_neg_wfn

        # Basis sets and wavefunctions from calculations with respect to magnetic field perturbations.
        self.mag_pos_basis = mag_pos_basis
        self.mag_neg_basis = mag_neg_basis
        self.mag_pos_wfn = mag_pos_wfn
        self.mag_neg_wfn = mag_neg_wfn

        # Components required for finite difference AATs.
        self.ao_overlap = []
        self.perturbed_density = []

    # Computes overlap between basis sets.
    def compute_ao_overlap(self):
        # Compute the overlap for positive displacements in nuclear position and the field, < Psi(R+) | Psi(R0, H+) >.
        for alpha in range(len(self.nuc_pos_basis)):
            for beta in range(len(self.mag_pos_basis)):

                # Setting the basis sets.
                nuc_basis_set = self.nuc_pos_basis[alpha]
                mag_basis_set = self.mag_pos_basis[beta]

                # Note: Setting the MintsHelper to either basis set does not make a difference.
                mints = psi4.core.MintsHelper(nuc_basis_set)

                # Computing the overlap.
                ao_overlap_pp = mints.ao_overlap(nuc_basis_set, mag_basis_set).np
                ao_palrevo_pp = mints.ao_overlap(mag_basis_set, nuc_basis_set).np
                # TEST
                print(ao_overlap_pp)
                print(ao_palrevo_pp)

                print("\n")


