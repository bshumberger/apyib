"""Contains the class and functions associated with computing the rotational strength for VCD calculations by finite difference at the Hartree-Fock level of theory."""

import psi4
import numpy as np
import math
import itertools as it
import time
from apyib.utils import run_psi4
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
#from apyib.finite_difference import finite_difference



# Computes the parity of a given list.
def perm_parity(a):
    parity = 1
    for i in range(0,len(a)-1):
        if a[i] != i:
            parity *= -1
            j = min(range(i,len(a)), key=a.__getitem__)
            a[i],a[j] = a[j],a[i]
    return parity



# Computes the molecular orbital overlap between two wavefunctions.
def compute_mo_overlap(ndocc, nbf, bra_basis, bra_wfn, ket_basis, ket_wfn):
    mints = psi4.core.MintsHelper(bra_basis)

    if bra_basis == ket_basis:
        ao_overlap = mints.ao_overlap().np
    elif bra_basis != ket_basis:
        ao_overlap = mints.ao_overlap(bra_basis, ket_basis).np

    mo_overlap = np.zeros_like(ao_overlap)
    mo_overlap = mo_overlap.astype('complex128')

    for m in range(0, nbf):
        for n in range(0, nbf):
            for mu in range(0, nbf):
                for nu in range(0, nbf):
                    mo_overlap[m, n] += np.conjugate(np.transpose(bra_wfn[mu, m])) *  ao_overlap[mu, nu] * ket_wfn[nu, n]
    return mo_overlap


# Computes the spin orbital overlap from the molecular orbital overlap.
def compute_so_overlap(nbf, mo_overlap):
    """ 
    Compute the spin orbital basis overlap matrix from the MO basis overlap matrix.
    """
    # Compute number of spin orbitals.
    nSO = 2 * nbf

    # Compute the SO Fock matrix.
    S_SO = np.zeros([nSO, nSO])
    S_SO = S_SO.astype('complex128')
    for p in range(0, nSO):
        if p % 2 == 0:
            p_spin = 1 
        elif p % 2 != 0:
            p_spin = -1
        for q in range(0, nSO):
            if q % 2 == 0:
                q_spin = 1 
            elif q % 2 != 0:
                q_spin = -1

            # Compute the spin integration.
            spin_int = p_spin * q_spin
            if spin_int < 0:
                spin_int = 0 

            # Compute spin orbital matrix elements.
            S_SO[p,q] = mo_overlap[p//2,q//2] * spin_int

    return S_SO



# Compute MO-level phase correction.
def compute_phase(ndocc, nbf, unperturbed_basis, unperturbed_wfn, ket_basis, ket_wfn):
    # Compute MO overlaps.
    mo_overlap1 = compute_mo_overlap(ndocc, nbf, unperturbed_basis, unperturbed_wfn, ket_basis, ket_wfn)
    mo_overlap2 = np.conjugate(np.transpose(mo_overlap1))

    new_ket_wfn = np.zeros_like(ket_wfn)

    # Compute the phase corrected coefficients.
    for m in range(0, nbf):
        # Compute the normalization.
        N = np.sqrt(mo_overlap1[m][m] * mo_overlap2[m][m])

        # Compute phase factor.
        phase_factor = mo_overlap1[m][m] / N 

        # Compute phase corrected overlap.
        for mu in range(0, nbf):
            new_ket_wfn[mu][m] = ket_wfn[mu][m] * (phase_factor ** -1)

    return new_ket_wfn



class AAT(object):
    """
    The atomic axial tensor object computed by finite difference.
    """
    def __init__(self, parameters, nbf, ndocc, unperturbed_wfn, unperturbed_basis, unperturbed_t2, nuc_pos_wfn, nuc_neg_wfn, nuc_pos_basis, nuc_neg_basis, nuc_pos_t2, nuc_neg_t2, mag_pos_wfn, mag_neg_wfn, mag_pos_basis, mag_neg_basis, mag_pos_t2, mag_neg_t2, nuc_pert_strength, mag_pert_strength):

        # Basis sets and wavefunctions from calculations with respect to nuclear displacements.
        self.nuc_pos_basis = nuc_pos_basis
        self.nuc_neg_basis = nuc_neg_basis
        self.nuc_pos_wfn = nuc_pos_wfn
        self.nuc_neg_wfn = nuc_neg_wfn
        self.nuc_pos_t2 = nuc_pos_t2
        self.nuc_neg_t2 = nuc_neg_t2

        # Basis sets and wavefunctions from calculations with respect to magnetic field perturbations.
        self.mag_pos_basis = mag_pos_basis
        self.mag_neg_basis = mag_neg_basis
        self.mag_pos_wfn = mag_pos_wfn
        self.mag_neg_wfn = mag_neg_wfn
        self.mag_pos_t2 = mag_pos_t2
        self.mag_neg_t2 = mag_neg_t2

        # Components required for finite difference AATs.
        self.nuc_pert_strength = nuc_pert_strength
        self.mag_pert_strength = mag_pert_strength

        # Components required for permutations.
        self.nbf = nbf
        self.ndocc = ndocc

        # Components required for unperturbed wavefunction.
        self.unperturbed_basis = unperturbed_basis
        self.unperturbed_wfn = unperturbed_wfn
        self.unperturbed_t2 = unperturbed_t2

    ## Computes the permutations required for the Hartree-Fock wavefunction.
    #def compute_perms(self):
    #    det = np.arange(0, self.ndocc)
    #    size = len(det)
    #    permutation = []
    #    parity = []

    #    heaperm(det, size, permutation, parity)
    #    return parity, permutation

    ## Computes the overlap between two Hartree-Fock wavefunctions.
    #def compute_hf_overlap(self, mo_overlap):
    #    det = np.arange(0, self.ndocc)
    #    mo_prod = 1
    #    hf_overlap = 0
    #    for n in it.permutations(det, r=None):
    #        perm = list(n)
    #        par = list(n)
    #        sign = perm_parity(par)
    #        for i in range(0, self.ndocc):
    #            mo_prod *= mo_overlap[det[i], perm[i]]
    #        hf_overlap += sign * mo_prod
    #        mo_prod = 1

    #    return hf_overlap

    # Computes the determinant of the occupied space involving a single row swap.
    def compute_det_1_row(self, mo_overlap, i, a):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving a single column swap.
    def compute_det_1_column(self, mo_overlap, k, c):
        S = mo_overlap.copy()
        S[:,[k,c]] = S[:,[c,k]]
        det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving a double row swap.
    def compute_det_2_row(self, mo_overlap, i, j, a, b):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[[j,b],:] = S[[b,j],:]
        if i == j:
            det_S = 0
        elif a == b:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving a double column swap.
    def compute_det_2_column(self, mo_overlap, k, l, c, d):
        S = mo_overlap.copy()
        S[:,[k,c]] = S[:,[c,k]]
        S[:,[l,d]] = S[:,[d,l]]
        if k == l:
            det_S = 0
        elif c == d:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S

    # Computes the determinant of the occupied space involving a row swap and a column swap.
    def compute_det_1_row_1_column(self, mo_overlap, i, a, k, c):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[:,[k,c]] = S[:,[c,k]]
        det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S

    # Computes the determinant of the occupied space involving two row swaps and a column swap.
    def compute_det_2_row_1_column(self, mo_overlap, i, j, a, b, k, c):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[[j,b],:] = S[[b,j],:]
        S[:,[k,c]] = S[:,[c,k]]
        if i == j:
            det_S = 0
        elif a == b:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving a row swap and two column swaps.
    def compute_det_1_row_2_column(self, mo_overlap, i, a, k, l, c, d):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[:,[k,c]] = S[:,[c,k]]
        S[:,[l,d]] = S[:,[d,l]]
        if k == l:
            det_S = 0
        elif c == d:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving two row swaps and two column swaps.
    def compute_det_2_row_2_column(self, mo_overlap, i, j, a, b, k, l, c, d):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[[j,b],:] = S[[b,j],:]
        S[:,[k,c]] = S[:,[c,k]]
        S[:,[l,d]] = S[:,[d,l]]
        if i == j:
            det_S = 0
        elif a == b:
            det_S = 0
        elif k == l:
            det_S = 0
        elif c == d:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S


    # Computes the determinant of the occupied space involving a single row swap.
    def compute_det_1_row_SO(self, mo_overlap, i, a):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving a single column swap.
    def compute_det_1_column_SO(self, mo_overlap, k, c):
        S = mo_overlap.copy()
        S[:,[k,c]] = S[:,[c,k]]
        det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving a double row swap.
    def compute_det_2_row_SO(self, mo_overlap, i, j, a, b):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[[j,b],:] = S[[b,j],:]
        if i == j:
            det_S = 0
        elif a == b:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving a double column swap.
    def compute_det_2_column_SO(self, mo_overlap, k, l, c, d):
        S = mo_overlap.copy()
        S[:,[k,c]] = S[:,[c,k]]
        S[:,[l,d]] = S[:,[d,l]]
        if k == l:
            det_S = 0
        elif c == d:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving a row swap and a column swap.
    def compute_det_1_row_1_column_SO(self, mo_overlap, i, a, k, c):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[:,[k,c]] = S[:,[c,k]]
        det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving two row swaps and a column swap.
    def compute_det_2_row_1_column_SO(self, mo_overlap, i, j, a, b, k, c):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[[j,b],:] = S[[b,j],:]
        S[:,[k,c]] = S[:,[c,k]]
        if i == j:
            det_S = 0
        elif a == b:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving a row swap and two column swaps.
    def compute_det_1_row_2_column_SO(self, mo_overlap, i, a, k, l, c, d):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[:,[k,c]] = S[:,[c,k]]
        S[:,[l,d]] = S[:,[d,l]]
        if k == l:
            det_S = 0
        elif c == d:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S

    # Computes the determinant of the occupied space involving two row swaps and two column swaps.
    def compute_det_2_row_2_column_SO(self, mo_overlap, i, j, a, b, k, l, c, d):
        S = mo_overlap.copy()
        S[[i,a],:] = S[[a,i],:]
        S[[j,b],:] = S[[b,j],:]
        S[:,[k,c]] = S[:,[c,k]]
        S[:,[l,d]] = S[:,[d,l]]
        if i == j:
            det_S = 0
        elif a == b:
            det_S = 0
        elif k == l:
            det_S = 0
        elif c == d:
            det_S = 0
        else:
            det_S = np.linalg.det(S[0:2*self.ndocc,0:2*self.ndocc])

        return det_S


    # Computes the Hartree-Fock AATs.
    def compute_hf_aat(self, alpha, beta):
        # Compute phase corrected wavefunctions.
        pc_nuc_pos_wfn = self.nuc_pos_wfn[alpha] #compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha])
        pc_nuc_neg_wfn = self.nuc_neg_wfn[alpha] #compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha])
        pc_mag_pos_wfn = self.mag_pos_wfn[beta] #compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        pc_mag_neg_wfn = self.mag_neg_wfn[beta] #compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_neg_basis[beta], self.mag_neg_wfn[beta])

        # Compute molecular orbital overlaps with phase correction applied.
        mo_overlap_pp = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], pc_nuc_pos_wfn, self.mag_pos_basis[beta], pc_mag_pos_wfn)
        mo_overlap_np = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], pc_nuc_neg_wfn, self.mag_pos_basis[beta], pc_mag_pos_wfn)
        mo_overlap_pn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], pc_nuc_pos_wfn, self.mag_neg_basis[beta], pc_mag_neg_wfn)
        mo_overlap_nn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], pc_nuc_neg_wfn, self.mag_neg_basis[beta], pc_mag_neg_wfn)

        # Compute Hartree-Fock overlaps.
        # The HF determinant in the spatial orbital basis needs to be squared to acount for the beta spin.
        hf_pp = np.linalg.det(mo_overlap_pp[0:self.ndocc, 0:self.ndocc])**2
        hf_np = np.linalg.det(mo_overlap_np[0:self.ndocc, 0:self.ndocc])**2
        hf_pn = np.linalg.det(mo_overlap_pn[0:self.ndocc, 0:self.ndocc])**2
        hf_nn = np.linalg.det(mo_overlap_nn[0:self.ndocc, 0:self.ndocc])**2

        # Compute the AAT.
        I = (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_pp - hf_np - hf_pn + hf_nn)

        return I



    # Computes the terms in the CID AATs with spin-orbitals. Note that the phase corrections were applied in the finite difference code.
    def compute_cid_aat_SO(self, alpha, beta):

        # Compute SO overlaps.
        # < psi | psi >
        mo_overlap_uu = compute_mo_overlap(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.unperturbed_basis, self.unperturbed_wfn)
        so_overlap_uu = compute_so_overlap(self.nbf, mo_overlap_uu)

        # < psi | dpsi/dH >
        mo_overlap_up = compute_mo_overlap(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        so_overlap_up = compute_so_overlap(self.nbf, mo_overlap_up)
        mo_overlap_un = compute_mo_overlap(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_neg_basis[beta], self.mag_neg_wfn[beta])
        so_overlap_un = compute_so_overlap(self.nbf, mo_overlap_un)

        # < dpsi/dR | psi >
        mo_overlap_pu = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], self.unperturbed_basis, self.unperturbed_wfn)
        so_overlap_pu = compute_so_overlap(self.nbf, mo_overlap_pu)
        mo_overlap_nu = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], self.unperturbed_basis, self.unperturbed_wfn)
        so_overlap_nu = compute_so_overlap(self.nbf, mo_overlap_nu)

        # < dpsi/dR | dpsi/dH >
        mo_overlap_pp = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        so_overlap_pp = compute_so_overlap(self.nbf, mo_overlap_pp)
        mo_overlap_pn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], self.mag_neg_basis[beta], self.mag_neg_wfn[beta])
        so_overlap_pn = compute_so_overlap(self.nbf, mo_overlap_pn)
        mo_overlap_np = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        so_overlap_np = compute_so_overlap(self.nbf, mo_overlap_np)
        mo_overlap_nn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], self.mag_neg_basis[beta], self.mag_neg_wfn[beta])
        so_overlap_nn = compute_so_overlap(self.nbf, mo_overlap_nn)

        # Compute Hartree-Fock overlaps.
        hf_pgpg = np.linalg.det(so_overlap_pp[0:2*self.ndocc, 0:2*self.ndocc])
        hf_ngpg = np.linalg.det(so_overlap_pn[0:2*self.ndocc, 0:2*self.ndocc])
        hf_pgng = np.linalg.det(so_overlap_np[0:2*self.ndocc, 0:2*self.ndocc])
        hf_ngng = np.linalg.det(so_overlap_nn[0:2*self.ndocc, 0:2*self.ndocc])

        # Compute normalization the normalization for each wavefunction.
        N_unperturbed = np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_t2), self.unperturbed_t2))
        N_nuc_pos = np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_t2[alpha]), self.nuc_pos_t2[alpha]))
        N_nuc_neg = np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_t2[alpha]), self.nuc_neg_t2[alpha]))
        N_mag_pos = np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_t2[beta]), self.mag_pos_t2[beta]))
        N_mag_neg = np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_t2[beta]), self.mag_neg_t2[beta]))

        #print(N_nuc_pos - N_nuc_neg, N_mag_pos - N_mag_neg, N_unperturbed)

        # Compute the HF term of the CID AAT.
        I = ( (1/(N_nuc_pos*N_mag_pos)) * hf_pgpg - (1/(N_nuc_pos*N_mag_neg)) * hf_pgng - (1/(N_nuc_neg*N_mag_pos)) * hf_ngpg + (1/(N_nuc_neg*N_mag_neg)) * hf_ngng)
        
        I = 0

        # Using spin-orbital formulation for MP2 and CID contribution to the AATs.
        ndocc = 2 * self.ndocc
        nbf = 2 * self.nbf


        # Compute the terms including only one doubly excited determinant in either the bra or ket. 
        for i in range(0, ndocc):
            for j in range(0, ndocc):
                for a in range(ndocc, nbf):
                    for b in range(ndocc, nbf):

                        # Compute determinant overlap.
                        # < ijab | d0/dH >
                        det_overlap_egp = self.compute_det_2_row_SO(so_overlap_up, i, j, a, b)
                        det_overlap_egn = self.compute_det_2_row_SO(so_overlap_un, i, j, a, b)
                        
                        # < dijab/dR | d0/dH >
                        det_overlap_epgp = self.compute_det_2_row_SO(so_overlap_pp, i, j, a, b)
                        det_overlap_epgn = self.compute_det_2_row_SO(so_overlap_pn, i, j, a, b)
                        det_overlap_engp = self.compute_det_2_row_SO(so_overlap_np, i, j, a, b)
                        det_overlap_engn = self.compute_det_2_row_SO(so_overlap_nn, i, j, a, b)

                        # < d0/dR | ijab >
                        det_overlap_gpe = self.compute_det_2_column_SO(so_overlap_pu, i, j, a, b)
                        det_overlap_gne = self.compute_det_2_column_SO(so_overlap_nu, i, j, a, b)

                        # < d0/dR | dijab/dH >
                        det_overlap_gpep = self.compute_det_2_column_SO(so_overlap_pp, i, j, a, b)
                        det_overlap_gpen = self.compute_det_2_column_SO(so_overlap_pn, i, j, a, b)
                        det_overlap_gnep = self.compute_det_2_column_SO(so_overlap_np, i, j, a, b)
                        det_overlap_gnen = self.compute_det_2_column_SO(so_overlap_nn, i, j, a, b)

                        # Compute contribution of this component to the AAT.
                        I += 0.25 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * ((1/(N_unperturbed*N_mag_pos)) * det_overlap_egp - (1/(N_unperturbed*N_mag_neg)) * det_overlap_egn)
                        I += 0.25 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * ((1/(N_nuc_pos*N_mag_pos)) * det_overlap_epgp - (1/(N_nuc_pos*N_mag_neg)) * det_overlap_epgn - (1/(N_nuc_neg*N_mag_pos)) * det_overlap_engp + (1/(N_nuc_neg*N_mag_neg)) * det_overlap_engn)
                        I += 0.25 * (self.mag_pos_t2[beta][i][j][a-ndocc][b-ndocc] - self.mag_neg_t2[beta][i][j][a-ndocc][b-ndocc]) * ((1/(N_nuc_pos*N_unperturbed)) * det_overlap_gpe - (1/(N_nuc_neg*N_unperturbed)) * det_overlap_gne)
                        I += 0.25 * self.unperturbed_t2[i][j][a-ndocc][b-ndocc] * ((1/(N_nuc_pos*N_mag_pos)) * det_overlap_gpep - (1/(N_nuc_pos*N_mag_neg)) * det_overlap_gpen - (1/(N_nuc_neg*N_mag_pos)) * det_overlap_gnep + (1/(N_nuc_neg*N_mag_neg)) * det_overlap_gnen)
 
                        # Compute the terms including the doubly excited determinant in the bra and ket. 
                        for k in range(0, ndocc):
                            for l in range(0, ndocc):
                                for c in range(ndocc, nbf):
                                    for d in range(ndocc, nbf):

                                        # Compute determinant overlap.
                                        # < ijab | klcd >
                                        det_overlap_ee = self.compute_det_2_row_2_column_SO(so_overlap_uu, i, j, a, b, k, l, c, d)

                                        # < ijab | dklcd/dH >
                                        det_overlap_eep = self.compute_det_2_row_2_column_SO(so_overlap_up, i, j, a, b, k, l, c, d)
                                        det_overlap_een = self.compute_det_2_row_2_column_SO(so_overlap_un, i, j, a, b, k, l, c, d)

                                        # < dijab/dR | klcd >
                                        det_overlap_epe = self.compute_det_2_row_2_column_SO(so_overlap_pu, i, j, a, b, k, l, c, d)
                                        det_overlap_ene = self.compute_det_2_row_2_column_SO(so_overlap_nu, i, j, a, b, k, l, c, d)

                                        # < dijab/dR | dklcd/dH >
                                        det_overlap_epep = self.compute_det_2_row_2_column_SO(so_overlap_pp, i, j, a, b, k, l, c, d)
                                        det_overlap_epen = self.compute_det_2_row_2_column_SO(so_overlap_pn, i, j, a, b, k, l, c, d)
                                        det_overlap_enep = self.compute_det_2_row_2_column_SO(so_overlap_np, i, j, a, b, k, l, c, d)
                                        det_overlap_enen = self.compute_det_2_row_2_column_SO(so_overlap_nn, i, j, a, b, k, l, c, d)

                                        # Compute contribution of this component to the AAT.
                                        I += 0.25**2 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * (self.mag_pos_t2[beta][k][l][c-ndocc][d-ndocc] - self.mag_neg_t2[beta][k][l][c-ndocc][d-ndocc]) * (1/(N_unperturbed**2)) * det_overlap_ee
                                        I += 0.25**2 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * self.unperturbed_t2[k][l][c-ndocc][d-ndocc] * ((1/(N_unperturbed*N_mag_pos)) * det_overlap_eep - (1/(N_unperturbed*N_mag_neg)) * det_overlap_een)
                                        I += 0.25**2 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * (self.mag_pos_t2[beta][k][l][c-ndocc][d-ndocc] - self.mag_neg_t2[beta][k][l][c-ndocc][d-ndocc]) * ((1/(N_nuc_pos*N_unperturbed)) * det_overlap_epe - (1/(N_nuc_neg*N_unperturbed)) * det_overlap_ene)
                                        I += 0.25**2 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * self.unperturbed_t2[k][l][c-ndocc][d-ndocc] * ((1/(N_nuc_pos*N_mag_pos)) * det_overlap_epep - (1/(N_nuc_pos*N_mag_neg)) * det_overlap_epen - (1/(N_nuc_neg*N_mag_pos)) * det_overlap_enep + (1/(N_nuc_neg*N_mag_neg)) * det_overlap_enen)

 
        return I * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength)



# Computes the terms in the CID AATs with spatial orbitals. Note that the phase corrections were applied in the finite difference code.
    def compute_cid_aat(self, alpha, beta):

        # Compute SO overlaps.
        # < psi | psi >
        mo_overlap_uu = compute_mo_overlap(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.unperturbed_basis, self.unperturbed_wfn)

        # < psi | dpsi/dH >
        mo_overlap_up = compute_mo_overlap(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        mo_overlap_un = compute_mo_overlap(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_neg_basis[beta], self.mag_neg_wfn[beta])

        # < dpsi/dR | psi >
        mo_overlap_pu = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], self.unperturbed_basis, self.unperturbed_wfn)
        mo_overlap_nu = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], self.unperturbed_basis, self.unperturbed_wfn)

        # < dpsi/dR | dpsi/dH >
        mo_overlap_pp = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        mo_overlap_pn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], self.mag_neg_basis[beta], self.mag_neg_wfn[beta])
        mo_overlap_np = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        mo_overlap_nn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], self.mag_neg_basis[beta], self.mag_neg_wfn[beta])

        # Compute Hartree-Fock overlaps.
        hf_pgpg = np.linalg.det(mo_overlap_pp[0:self.ndocc, 0:self.ndocc])
        hf_ngpg = np.linalg.det(mo_overlap_pn[0:self.ndocc, 0:self.ndocc])
        hf_pgng = np.linalg.det(mo_overlap_np[0:self.ndocc, 0:self.ndocc])
        hf_ngng = np.linalg.det(mo_overlap_nn[0:self.ndocc, 0:self.ndocc])

        # Compute normalization the normalization for each wavefunction.
        N_unperturbed = np.sqrt(1 + 0.5**2 * (2*np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_t2), self.unperturbed_t2) - np.einsum('ijab,ijba->', np.conjugate(self.unperturbed_t2), self.unperturbed_t2)))
        N_nuc_pos = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_t2[alpha]), self.nuc_pos_t2[alpha]))
        N_nuc_neg = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_t2[alpha]), self.nuc_neg_t2[alpha]))
        N_mag_pos = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_t2[beta]), self.mag_pos_t2[beta]))
        N_mag_neg = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_t2[beta]), self.mag_neg_t2[beta]))

        N_nuc_pos = np.sqrt(1 + 0.5**2 * (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_t2[alpha]), self.nuc_pos_t2[alpha]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_pos_t2[alpha]), self.nuc_pos_t2[alpha])))
        N_nuc_neg = np.sqrt(1 + 0.5**2 * (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_t2[alpha]), self.nuc_neg_t2[alpha]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_neg_t2[alpha]), self.nuc_neg_t2[alpha])))
        N_mag_pos = np.sqrt(1 + 0.5**2 * (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_t2[beta]), self.mag_pos_t2[beta]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_pos_t2[beta]), self.mag_pos_t2[beta])))
        N_mag_neg = np.sqrt(1 + 0.5**2 * (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_t2[beta]), self.mag_neg_t2[beta]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_neg_t2[beta]), self.mag_neg_t2[beta])))

        #print(N_nuc_pos - N_nuc_neg, N_mag_pos - N_mag_neg, N_unperturbed)


        # Compute the HF term of the CID AAT.
        I = (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * ( (1/(N_nuc_pos*N_mag_pos)) * hf_pgpg - (1/(N_nuc_pos*N_mag_neg)) * hf_pgng - (1/(N_nuc_neg*N_mag_pos)) * hf_ngpg + (1/(N_nuc_neg*N_mag_neg)) * hf_ngng)
    
        I = 0

        # Compute the terms including only one doubly excited determinant in either the bra or ket. 
        for i in range(0, self.ndocc):
            for j in range(0, self.ndocc):
                for a in range(self.ndocc, self.nbf):
                    for b in range(self.ndocc, self.nbf):

                        # Swap the rows for orbital substituion in the bra.
                        # < ijab | d0/dH >
                        # Unperturbed/Positive
                        S_up = mo_overlap_up.copy()
                        det_S_up = np.linalg.det(S_up[0:self.ndocc,0:self.ndocc])
                        det_ijab_S_up = self.compute_det_2_row(mo_overlap_up, i, j, a, b)
                        det_ia_S_up = self.compute_det_1_row(mo_overlap_up, i, a)
                        det_jb_S_up = self.compute_det_1_row(mo_overlap_up, j, b)
                        det_ib_S_up = self.compute_det_1_row(mo_overlap_up, i, b)
                        det_ja_S_up = self.compute_det_1_row(mo_overlap_up, j, a)

                        t_ijab_p = self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_p = self.nuc_pos_t2[alpha][i][j][b-self.ndocc][a-self.ndocc]

                        # Unperturbed/Negative
                        S_un = mo_overlap_un.copy()
                        det_S_un = np.linalg.det(S_un[0:self.ndocc,0:self.ndocc])
                        det_ijab_S_un = self.compute_det_2_row(mo_overlap_un, i, j, a, b)
                        det_ia_S_un = self.compute_det_1_row(mo_overlap_un, i, a)
                        det_jb_S_un = self.compute_det_1_row(mo_overlap_un, j, b)
                        det_ib_S_un = self.compute_det_1_row(mo_overlap_un, i, b)
                        det_ja_S_un = self.compute_det_1_row(mo_overlap_un, j, a)

                        t_ijab_n = self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_n = self.nuc_neg_t2[alpha][i][j][b-self.ndocc][a-self.ndocc]

                        I += 0.5 * np.conjugate((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * ((1/(N_unperturbed*N_mag_pos)) * det_ijab_S_up * det_S_up  - (1/(N_unperturbed*N_mag_neg)) * det_ijab_S_un * det_S_un)
                        I += 0.5 * 2 * np.conjugate((t_ijab_p - t_ijab_n)) * ((1/(N_unperturbed*N_mag_pos)) * det_ia_S_up * det_jb_S_up  - (1/(N_unperturbed*N_mag_neg)) * det_ia_S_un * det_jb_S_un)

                        # < dijab/dR | d0/dH >
                        # Positive/Positive
                        S_pp = mo_overlap_pp.copy()
                        det_S_pp = np.linalg.det(S_pp[0:self.ndocc,0:self.ndocc])
                        det_ijab_S_pp = self.compute_det_2_row(mo_overlap_pp, i, j, a, b)
                        det_ia_S_pp = self.compute_det_1_row(mo_overlap_pp, i, a)
                        det_jb_S_pp = self.compute_det_1_row(mo_overlap_pp, j, b)
                        det_ib_S_pp = self.compute_det_1_row(mo_overlap_pp, i, b)
                        det_ja_S_pp = self.compute_det_1_row(mo_overlap_pp, j, a)

                        # Negative/Negative
                        S_nn = mo_overlap_nn.copy()
                        det_S_nn = np.linalg.det(S_nn[0:self.ndocc,0:self.ndocc])
                        det_ijab_S_nn = self.compute_det_2_row(mo_overlap_nn, i, j, a, b)
                        det_ia_S_nn = self.compute_det_1_row(mo_overlap_nn, i, a)
                        det_jb_S_nn = self.compute_det_1_row(mo_overlap_nn, j, b)
                        det_ib_S_nn = self.compute_det_1_row(mo_overlap_nn, i, b)
                        det_ja_S_nn = self.compute_det_1_row(mo_overlap_nn, j, a)

                        # Positive/Negative
                        S_pn = mo_overlap_pn.copy()
                        det_S_pn = np.linalg.det(S_pn[0:self.ndocc,0:self.ndocc])
                        det_ijab_S_pn = self.compute_det_2_row(mo_overlap_pn, i, j, a, b)
                        det_ia_S_pn = self.compute_det_1_row(mo_overlap_pn, i, a)
                        det_jb_S_pn = self.compute_det_1_row(mo_overlap_pn, j, b)
                        det_ib_S_pn = self.compute_det_1_row(mo_overlap_pn, i, b)
                        det_ja_S_pn = self.compute_det_1_row(mo_overlap_pn, j, a)

                        # Negative/Positive
                        S_np = mo_overlap_np.copy()
                        det_S_np = np.linalg.det(S_np[0:self.ndocc,0:self.ndocc])
                        det_ijab_S_np = self.compute_det_2_row(mo_overlap_np, i, j, a, b)
                        det_ia_S_np = self.compute_det_1_row(mo_overlap_np, i, a)
                        det_jb_S_np = self.compute_det_1_row(mo_overlap_np, j, b)
                        det_ib_S_np = self.compute_det_1_row(mo_overlap_np, i, b)
                        det_ja_S_np = self.compute_det_1_row(mo_overlap_np, j, a)

                        # Amplitudes
                        t_ijab = self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba = self.unperturbed_t2[i][j][b-self.ndocc][a-self.ndocc]

                        I += 0.5 * np.conjugate((t_ijab - t_ijba)) * ((1/(N_nuc_pos*N_mag_pos)) * det_ijab_S_pp * det_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_ijab_S_pn * det_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ijab_S_np * det_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ijab_S_nn * det_S_nn)
                        I += 0.5 * 2 * np.conjugate(t_ijab) * ((1/(N_nuc_pos*N_mag_pos)) * det_ia_S_pp * det_jb_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_ia_S_pn * det_jb_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ia_S_np * det_jb_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ia_S_nn * det_jb_S_nn)

                        # Swap the columns for orbital substituion in the ket.
                        # < d0/dR | ijab >
                        # Positive
                        S_pu = mo_overlap_pu.copy()
                        det_S_pu = np.linalg.det(S_pu[0:self.ndocc,0:self.ndocc])
                        det_S_ijab_pu = self.compute_det_2_column(mo_overlap_pu, i, j, a, b)
                        det_S_ia_pu = self.compute_det_1_column(mo_overlap_pu, i, a)
                        det_S_jb_pu = self.compute_det_1_column(mo_overlap_pu, j, b)
                        det_S_ib_pu = self.compute_det_1_column(mo_overlap_pu, i, b)
                        det_S_ja_pu = self.compute_det_1_column(mo_overlap_pu, j, a)

                        t_ijab_p = self.mag_pos_t2[beta][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_p = self.mag_pos_t2[beta][i][j][b-self.ndocc][a-self.ndocc]

                        # Negative
                        S_nu = mo_overlap_nu.copy()
                        det_S_nu = np.linalg.det(S_nu[0:self.ndocc,0:self.ndocc])
                        det_S_ijab_nu = self.compute_det_2_column(mo_overlap_nu, i, j, a, b)
                        det_S_ia_nu = self.compute_det_1_column(mo_overlap_nu, i, a)
                        det_S_jb_nu = self.compute_det_1_column(mo_overlap_nu, j, b)
                        det_S_ib_nu = self.compute_det_1_column(mo_overlap_nu, i, b)
                        det_S_ja_nu = self.compute_det_1_column(mo_overlap_nu, j, a)

                        t_ijab_n = self.mag_neg_t2[beta][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_n = self.mag_neg_t2[beta][i][j][b-self.ndocc][a-self.ndocc]

                        I += 0.5 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * ((1/(N_unperturbed*N_nuc_pos)) * det_S_ijab_pu * det_S_pu  - (1/(N_unperturbed*N_nuc_neg)) * det_S_ijab_nu * det_S_nu)
                        I += 0.5 * 2 * (t_ijab_p - t_ijab_n) * ((1/(N_unperturbed*N_nuc_pos)) * det_S_ia_pu * det_S_jb_pu  - (1/(N_unperturbed*N_nuc_neg)) * det_S_ia_nu * det_S_jb_nu)

                        # < d0/dR | dijab/dH >
                        # Positive/Positive
                        S_pp = mo_overlap_pp.copy()
                        det_S_pp = np.linalg.det(S_pp[0:self.ndocc,0:self.ndocc])
                        det_S_ijab_pp = self.compute_det_2_column(mo_overlap_pp, i, j, a, b)
                        det_S_ia_pp = self.compute_det_1_column(mo_overlap_pp, i, a)
                        det_S_jb_pp = self.compute_det_1_column(mo_overlap_pp, j, b)
                        det_S_ib_pp = self.compute_det_1_column(mo_overlap_pp, i, b)
                        det_S_ja_pp = self.compute_det_1_column(mo_overlap_pp, j, a)

                        # Negative/Negative
                        S_nn = mo_overlap_nn.copy()
                        det_S_nn = np.linalg.det(S_nn[0:self.ndocc,0:self.ndocc])
                        det_S_ijab_nn = self.compute_det_2_column(mo_overlap_nn, i, j, a, b)
                        det_S_ia_nn = self.compute_det_1_column(mo_overlap_nn, i, a)
                        det_S_jb_nn = self.compute_det_1_column(mo_overlap_nn, j, b)
                        det_S_ib_nn = self.compute_det_1_column(mo_overlap_nn, i, b)
                        det_S_ja_nn = self.compute_det_1_column(mo_overlap_nn, j, a)

                        # Positive/Negative
                        S_pn = mo_overlap_pn.copy()
                        det_S_pn = np.linalg.det(S_pn[0:self.ndocc,0:self.ndocc])
                        det_S_ijab_pn = self.compute_det_2_column(mo_overlap_pn, i, j, a, b)
                        det_S_ia_pn = self.compute_det_1_column(mo_overlap_pn, i, a)
                        det_S_jb_pn = self.compute_det_1_column(mo_overlap_pn, j, b)
                        det_S_ib_pn = self.compute_det_1_column(mo_overlap_pn, i, b)
                        det_S_ja_pn = self.compute_det_1_column(mo_overlap_pn, j, a)

                        # Negative/Positive
                        S_np = mo_overlap_np.copy()
                        det_S_np = np.linalg.det(S_np[0:self.ndocc,0:self.ndocc])
                        det_S_ijab_np = self.compute_det_2_column(mo_overlap_np, i, j, a, b)
                        det_S_ia_np = self.compute_det_1_column(mo_overlap_np, i, a)
                        det_S_jb_np = self.compute_det_1_column(mo_overlap_np, j, b)
                        det_S_ib_np = self.compute_det_1_column(mo_overlap_np, i, b)
                        det_S_ja_np = self.compute_det_1_column(mo_overlap_np, j, a)

                        # Amplitudes
                        t_ijab = self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba = self.unperturbed_t2[i][j][b-self.ndocc][a-self.ndocc]

                        I += 0.5 * (t_ijab - t_ijba) * ((1/(N_nuc_pos*N_mag_pos)) * det_S_ijab_pp * det_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_S_ijab_pn * det_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_S_ijab_np * det_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_S_ijab_nn * det_S_nn)
                        I += 0.5 * 2 * (t_ijab) * ((1/(N_nuc_pos*N_mag_pos)) * det_S_ia_pp * det_S_jb_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_S_ia_pn * det_S_jb_pn - (1/(N_nuc_neg*N_mag_pos)) * det_S_ia_np * det_S_jb_np + (1/(N_nuc_neg*N_mag_neg)) * det_S_ia_nn * det_S_jb_nn)

                        for k in range(0, self.ndocc):
                            for l in range(0, self.ndocc):
                                for c in range(self.ndocc, self.nbf):
                                    for d in range(self.ndocc, self.nbf):

                                        # Amplitudes
                                        t_ijab_p = np.conjugate(self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc])
                                        t_ijba_p = np.conjugate(self.nuc_pos_t2[alpha][i][j][b-self.ndocc][a-self.ndocc])
                                        t_ijab_n = np.conjugate(self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc])
                                        t_ijba_n = np.conjugate(self.nuc_neg_t2[alpha][i][j][b-self.ndocc][a-self.ndocc])

                                        t_klcd_p = self.mag_pos_t2[beta][k][l][c-self.ndocc][d-self.ndocc]
                                        t_kldc_p = self.mag_pos_t2[beta][k][l][d-self.ndocc][c-self.ndocc]
                                        t_klcd_n = self.mag_neg_t2[beta][k][l][c-self.ndocc][d-self.ndocc]
                                        t_kldc_n = self.mag_neg_t2[beta][k][l][d-self.ndocc][c-self.ndocc]

                                        t_ijab = np.conjugate(self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc])
                                        t_ijba = np.conjugate(self.unperturbed_t2[i][j][b-self.ndocc][a-self.ndocc])
                                        t_klcd = self.unperturbed_t2[k][l][c-self.ndocc][d-self.ndocc]
                                        t_kldc = self.unperturbed_t2[k][l][d-self.ndocc][c-self.ndocc]

                                        # < ijab | klcd >
                                        # Unperturbed/Unperturbed
                                        S_uu = mo_overlap_uu.copy()
                                        det_S_uu = np.linalg.det(S_uu[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_uu = self.compute_det_2_row_2_column(mo_overlap_uu, i, j, a, b, k, l, c, d)
                                        det_ijab_S_uu = self.compute_det_2_row(mo_overlap_uu, i, j, a, b)
                                        det_S_klcd_uu = self.compute_det_2_column(mo_overlap_uu, k, l, c, d)
                                        det_ijab_S_kc_uu = self.compute_det_2_row_1_column(mo_overlap_uu, i, j, a, b, k, c)
                                        det_S_ld_uu = self.compute_det_1_column(mo_overlap_uu, l, d)
                                        det_ijab_S_kd_uu = self.compute_det_2_row_1_column(mo_overlap_uu, i, j, a, b, k, d)
                                        det_S_lc_uu = self.compute_det_1_column(mo_overlap_uu, l, c)

                                        det_ia_S_klcd_uu = self.compute_det_1_row_2_column(mo_overlap_uu, i, a, k, l, c, d)
                                        det_jb_S_uu = self.compute_det_1_row(mo_overlap_uu, j, b)
                                        det_ia_S_uu = self.compute_det_1_row(mo_overlap_uu, i, a)
                                        det_jb_S_klcd_uu = self.compute_det_1_row_2_column(mo_overlap_uu, j, b, k, l, c, d)
                                        det_ia_S_kc_uu = self.compute_det_1_row_1_column(mo_overlap_uu, i, a, k, c)
                                        det_jb_S_ld_uu = self.compute_det_1_row_1_column(mo_overlap_uu, j, b, l, d)
                                        det_ia_S_kd_uu = self.compute_det_1_row_1_column(mo_overlap_uu, i, a, k, d)
                                        det_jb_S_lc_uu = self.compute_det_1_row_1_column(mo_overlap_uu, j, b, l, c)

                                        I += 0.125 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * (1/(N_unperturbed**2)) * det_ijab_S_klcd_uu * det_S_uu
                                        I += 0.125 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * (1/(N_unperturbed**2)) * det_ijab_S_uu * det_S_klcd_uu
                                        I += 0.125 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * 4 * (t_klcd_p - t_klcd_n) * (1/(N_unperturbed**2)) * det_ijab_S_kc_uu * det_S_ld_uu

                                        I += 0.125 * 2 * (t_ijab_p - t_ijab_n) * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * (1/(N_unperturbed**2)) * det_ia_S_klcd_uu * det_jb_S_uu
                                        I += 0.125 * 2 * (t_ijab_p - t_ijab_n) * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * (1/(N_unperturbed**2)) * det_ia_S_uu * det_jb_S_klcd_uu
                                        I += 0.125 * 2 * (t_ijab_p - t_ijab_n) * 4 * (t_klcd_p - t_klcd_n) * (1/(N_unperturbed**2)) * det_ia_S_kc_uu * det_jb_S_ld_uu

                                        # < ijab | dklcd/dH >
                                        # Unperturbed/Positive
                                        S_up = mo_overlap_up.copy()
                                        det_S_up = np.linalg.det(S_up[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_up = self.compute_det_2_row_2_column(mo_overlap_up, i, j, a, b, k, l, c, d)
                                        det_ijab_S_up = self.compute_det_2_row(mo_overlap_up, i, j, a, b)
                                        det_S_klcd_up = self.compute_det_2_column(mo_overlap_up, k, l, c, d)
                                        det_ijab_S_kc_up = self.compute_det_2_row_1_column(mo_overlap_up, i, j, a, b, k, c)
                                        det_S_ld_up = self.compute_det_1_column(mo_overlap_up, l, d)
                                        det_ijab_S_kd_up = self.compute_det_2_row_1_column(mo_overlap_up, i, j, a, b, k, d)
                                        det_S_lc_up = self.compute_det_1_column(mo_overlap_up, l, c)

                                        det_ia_S_klcd_up = self.compute_det_1_row_2_column(mo_overlap_up, i, a, k, l, c, d)
                                        det_jb_S_up = self.compute_det_1_row(mo_overlap_up, j, b)
                                        det_ia_S_up = self.compute_det_1_row(mo_overlap_up, i, a)
                                        det_jb_S_klcd_up = self.compute_det_1_row_2_column(mo_overlap_up, j, b, k, l, c, d)
                                        det_ia_S_kc_up = self.compute_det_1_row_1_column(mo_overlap_up, i, a, k, c)
                                        det_jb_S_ld_up = self.compute_det_1_row_1_column(mo_overlap_up, j, b, l, d)
                                        det_ia_S_kd_up = self.compute_det_1_row_1_column(mo_overlap_up, i, a, k, d)
                                        det_jb_S_lc_up = self.compute_det_1_row_1_column(mo_overlap_up, j, b, l, c)

                                        # Unperturbed/Negative
                                        S_un = mo_overlap_un.copy()
                                        det_S_un = np.linalg.det(S_un[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_un = self.compute_det_2_row_2_column(mo_overlap_un, i, j, a, b, k, l, c, d)
                                        det_ijab_S_un = self.compute_det_2_row(mo_overlap_un, i, j, a, b)
                                        det_S_klcd_un = self.compute_det_2_column(mo_overlap_un, k, l, c, d)
                                        det_ijab_S_kc_un = self.compute_det_2_row_1_column(mo_overlap_un, i, j, a, b, k, c)
                                        det_S_ld_un = self.compute_det_1_column(mo_overlap_un, l, d)
                                        det_ijab_S_kd_un = self.compute_det_2_row_1_column(mo_overlap_un, i, j, a, b, k, d)
                                        det_S_lc_un = self.compute_det_1_column(mo_overlap_un, l, c)

                                        det_ia_S_klcd_un = self.compute_det_1_row_2_column(mo_overlap_un, i, a, k, l, c, d)
                                        det_jb_S_un = self.compute_det_1_row(mo_overlap_un, j, b)
                                        det_ia_S_un = self.compute_det_1_row(mo_overlap_un, i, a)
                                        det_jb_S_klcd_un = self.compute_det_1_row_2_column(mo_overlap_un, j, b, k, l, c, d)
                                        det_ia_S_kc_un = self.compute_det_1_row_1_column(mo_overlap_un, i, a, k, c)
                                        det_jb_S_ld_un = self.compute_det_1_row_1_column(mo_overlap_un, j, b, l, d)
                                        det_ia_S_kd_un = self.compute_det_1_row_1_column(mo_overlap_un, i, a, k, d)
                                        det_jb_S_lc_un = self.compute_det_1_row_1_column(mo_overlap_un, j, b, l, c)

                                        I += 0.125 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * (t_klcd - t_kldc) * ((1/(N_unperturbed*N_mag_pos)) * det_ijab_S_klcd_up * det_S_up - (1/(N_unperturbed*N_mag_neg)) * det_ijab_S_klcd_un * det_S_un) 
                                        I += 0.125 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * (t_klcd - t_kldc) * ((1/(N_unperturbed*N_mag_pos)) * det_ijab_S_up * det_S_klcd_up - (1/(N_unperturbed*N_mag_neg)) * det_ijab_S_un * det_S_klcd_un)
                                        I += 0.125 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * 4 * t_klcd * ((1/(N_unperturbed*N_mag_pos)) * det_ijab_S_kc_up * det_S_ld_up - (1/(N_unperturbed*N_mag_neg)) * det_ijab_S_kc_un * det_S_ld_un) 


                                        I += 0.125 * 2 * (t_ijab_p - t_ijab_n) * (t_klcd - t_kldc) * ((1/(N_unperturbed*N_mag_pos)) * det_ia_S_klcd_up * det_jb_S_up - (1/(N_unperturbed*N_mag_neg)) * det_ia_S_klcd_un * det_jb_S_un)
                                        I += 0.125 * 2 * (t_ijab_p - t_ijab_n) * (t_klcd - t_kldc) * ((1/(N_unperturbed*N_mag_pos)) * det_ia_S_up * det_jb_S_klcd_up - (1/(N_unperturbed*N_mag_neg)) * det_ia_S_un * det_jb_S_klcd_un)
                                        I += 0.125 * 2 * (t_ijab_p - t_ijab_n) * 4 * t_klcd * ((1/(N_unperturbed*N_mag_pos)) * det_ia_S_kc_up * det_jb_S_ld_up - (1/(N_unperturbed*N_mag_neg)) * det_ia_S_kc_un * det_jb_S_ld_un)

                                        ## < dijab/dR | klcd >
                                        # Positive/Unperturbed
                                        S_pu = mo_overlap_pu.copy()
                                        det_S_pu = np.linalg.det(S_pu[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_pu = self.compute_det_2_row_2_column(mo_overlap_pu, i, j, a, b, k, l, c, d)
                                        det_ijab_S_pu = self.compute_det_2_row(mo_overlap_pu, i, j, a, b)
                                        det_S_klcd_pu = self.compute_det_2_column(mo_overlap_pu, k, l, c, d)
                                        det_ijab_S_kc_pu = self.compute_det_2_row_1_column(mo_overlap_pu, i, j, a, b, k, c)
                                        det_S_ld_pu = self.compute_det_1_column(mo_overlap_pu, l, d)
                                        det_ijab_S_kd_pu = self.compute_det_2_row_1_column(mo_overlap_pu, i, j, a, b, k, d)
                                        det_S_lc_pu = self.compute_det_1_column(mo_overlap_pu, l, c)

                                        det_ia_S_klcd_pu = self.compute_det_1_row_2_column(mo_overlap_pu, i, a, k, l, c, d)
                                        det_jb_S_pu = self.compute_det_1_row(mo_overlap_pu, j, b)
                                        det_ia_S_pu = self.compute_det_1_row(mo_overlap_pu, i, a)
                                        det_jb_S_klcd_pu = self.compute_det_1_row_2_column(mo_overlap_pu, j, b, k, l, c, d)
                                        det_ia_S_kc_pu = self.compute_det_1_row_1_column(mo_overlap_pu, i, a, k, c)
                                        det_jb_S_ld_pu = self.compute_det_1_row_1_column(mo_overlap_pu, j, b, l, d)
                                        det_ia_S_kd_pu = self.compute_det_1_row_1_column(mo_overlap_pu, i, a, k, d)
                                        det_jb_S_lc_pu = self.compute_det_1_row_1_column(mo_overlap_pu, j, b, l, c)

                                        # Negative/Unperturbed
                                        S_nu = mo_overlap_nu.copy()
                                        det_S_nu = np.linalg.det(S_nu[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_nu = self.compute_det_2_row_2_column(mo_overlap_nu, i, j, a, b, k, l, c, d)
                                        det_ijab_S_nu = self.compute_det_2_row(mo_overlap_nu, i, j, a, b)
                                        det_S_klcd_nu = self.compute_det_2_column(mo_overlap_nu, k, l, c, d)
                                        det_ijab_S_kc_nu = self.compute_det_2_row_1_column(mo_overlap_nu, i, j, a, b, k, c)
                                        det_S_ld_nu = self.compute_det_1_column(mo_overlap_nu, l, d)
                                        det_ijab_S_kd_nu = self.compute_det_2_row_1_column(mo_overlap_nu, i, j, a, b, k, d)
                                        det_S_lc_nu = self.compute_det_1_column(mo_overlap_nu, l, c)

                                        det_ia_S_klcd_nu = self.compute_det_1_row_2_column(mo_overlap_nu, i, a, k, l, c, d)
                                        det_jb_S_nu = self.compute_det_1_row(mo_overlap_nu, j, b)
                                        det_ia_S_nu = self.compute_det_1_row(mo_overlap_nu, i, a)
                                        det_jb_S_klcd_nu = self.compute_det_1_row_2_column(mo_overlap_nu, j, b, k, l, c, d)
                                        det_ia_S_kc_nu = self.compute_det_1_row_1_column(mo_overlap_nu, i, a, k, c)
                                        det_jb_S_ld_nu = self.compute_det_1_row_1_column(mo_overlap_nu, j, b, l, d)
                                        det_ia_S_kd_nu = self.compute_det_1_row_1_column(mo_overlap_nu, i, a, k, d)
                                        det_jb_S_lc_nu = self.compute_det_1_row_1_column(mo_overlap_nu, j, b, l, c)

                                        I += 0.125 * (t_ijab - t_ijba) * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed*N_nuc_pos)) * det_ijab_S_klcd_pu * det_S_pu - (1/(N_unperturbed*N_nuc_neg)) * det_ijab_S_klcd_nu * det_S_nu)
                                        I += 0.125 * (t_ijab - t_ijba) * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed*N_nuc_pos)) * det_ijab_S_pu * det_S_klcd_pu - (1/(N_unperturbed*N_nuc_neg)) * det_ijab_S_nu * det_S_klcd_nu)
                                        I += 0.125 * (t_ijab - t_ijba) * 4 * (t_klcd_p - t_klcd_n) * ((1/(N_unperturbed*N_nuc_pos)) * det_ijab_S_kc_pu * det_S_ld_pu - (1/(N_unperturbed*N_nuc_neg)) * det_ijab_S_kc_nu * det_S_ld_nu)


                                        I += 0.125 * 2 * t_ijab * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed*N_nuc_pos)) * det_ia_S_klcd_pu * det_jb_S_pu - (1/(N_unperturbed*N_nuc_neg)) * det_ia_S_klcd_nu * det_jb_S_nu)
                                        I += 0.125 * 2 * t_ijab * ((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed*N_nuc_pos)) * det_ia_S_pu * det_jb_S_klcd_pu - (1/(N_unperturbed*N_nuc_neg)) * det_ia_S_nu * det_jb_S_klcd_nu)
                                        I += 0.125 * 2 * t_ijab * 4 * (t_klcd_p - t_klcd_n) * ((1/(N_unperturbed*N_nuc_pos)) * det_ia_S_kc_pu * det_jb_S_ld_pu - (1/(N_unperturbed*N_nuc_neg)) * det_ia_S_kc_nu * det_jb_S_ld_nu) 

                                        # < dijab/dR | dklcd/dH >
                                        # Positive/Positive
                                        S_pp = mo_overlap_pp.copy()
                                        det_S_pp = np.linalg.det(S_pp[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_pp = self.compute_det_2_row_2_column(mo_overlap_pp, i, j, a, b, k, l, c, d)
                                        det_ijab_S_pp = self.compute_det_2_row(mo_overlap_pp, i, j, a, b)
                                        det_S_klcd_pp = self.compute_det_2_column(mo_overlap_pp, k, l, c, d)
                                        det_ijab_S_kc_pp = self.compute_det_2_row_1_column(mo_overlap_pp, i, j, a, b, k, c)
                                        det_S_ld_pp = self.compute_det_1_column(mo_overlap_pp, l, d)
                                        det_ijab_S_kd_pp = self.compute_det_2_row_1_column(mo_overlap_pp, i, j, a, b, k, d)
                                        det_S_lc_pp = self.compute_det_1_column(mo_overlap_pp, l, c)

                                        det_ia_S_klcd_pp = self.compute_det_1_row_2_column(mo_overlap_pp, i, a, k, l, c, d)
                                        det_jb_S_pp = self.compute_det_1_row(mo_overlap_pp, j, b)
                                        det_ia_S_pp = self.compute_det_1_row(mo_overlap_pp, i, a)
                                        det_jb_S_klcd_pp = self.compute_det_1_row_2_column(mo_overlap_pp, j, b, k, l, c, d)
                                        det_ia_S_kc_pp = self.compute_det_1_row_1_column(mo_overlap_pp, i, a, k, c)
                                        det_jb_S_ld_pp = self.compute_det_1_row_1_column(mo_overlap_pp, j, b, l, d)
                                        det_ia_S_kd_pp = self.compute_det_1_row_1_column(mo_overlap_pp, i, a, k, d)
                                        det_jb_S_lc_pp = self.compute_det_1_row_1_column(mo_overlap_pp, j, b, l, c)

                                        # Positive/Negative
                                        S_pn = mo_overlap_pn.copy()
                                        det_S_pn = np.linalg.det(S_pn[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_pn = self.compute_det_2_row_2_column(mo_overlap_pn, i, j, a, b, k, l, c, d)
                                        det_ijab_S_pn = self.compute_det_2_row(mo_overlap_pn, i, j, a, b)
                                        det_S_klcd_pn = self.compute_det_2_column(mo_overlap_pn, k, l, c, d)
                                        det_ijab_S_kc_pn = self.compute_det_2_row_1_column(mo_overlap_pn, i, j, a, b, k, c)
                                        det_S_ld_pn = self.compute_det_1_column(mo_overlap_pn, l, d)
                                        det_ijab_S_kd_pn = self.compute_det_2_row_1_column(mo_overlap_pn, i, j, a, b, k, d)
                                        det_S_lc_pn = self.compute_det_1_column(mo_overlap_pn, l, c)

                                        det_ia_S_klcd_pn = self.compute_det_1_row_2_column(mo_overlap_pn, i, a, k, l, c, d)
                                        det_jb_S_pn = self.compute_det_1_row(mo_overlap_pn, j, b)
                                        det_ia_S_pn = self.compute_det_1_row(mo_overlap_pn, i, a)
                                        det_jb_S_klcd_pn = self.compute_det_1_row_2_column(mo_overlap_pn, j, b, k, l, c, d)
                                        det_ia_S_kc_pn = self.compute_det_1_row_1_column(mo_overlap_pn, i, a, k, c)
                                        det_jb_S_ld_pn = self.compute_det_1_row_1_column(mo_overlap_pn, j, b, l, d)
                                        det_ia_S_kd_pn = self.compute_det_1_row_1_column(mo_overlap_pn, i, a, k, d)
                                        det_jb_S_lc_pn = self.compute_det_1_row_1_column(mo_overlap_pn, j, b, l, c)

                                        # Negative/Positive
                                        S_np = mo_overlap_np.copy()
                                        det_S_np = np.linalg.det(S_np[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_np = self.compute_det_2_row_2_column(mo_overlap_np, i, j, a, b, k, l, c, d)
                                        det_ijab_S_np = self.compute_det_2_row(mo_overlap_np, i, j, a, b)
                                        det_S_klcd_np = self.compute_det_2_column(mo_overlap_np, k, l, c, d)
                                        det_ijab_S_kc_np = self.compute_det_2_row_1_column(mo_overlap_np, i, j, a, b, k, c)
                                        det_S_ld_np = self.compute_det_1_column(mo_overlap_np, l, d)
                                        det_ijab_S_kd_np = self.compute_det_2_row_1_column(mo_overlap_np, i, j, a, b, k, d)
                                        det_S_lc_np = self.compute_det_1_column(mo_overlap_np, l, c)

                                        det_ia_S_klcd_np = self.compute_det_1_row_2_column(mo_overlap_np, i, a, k, l, c, d)
                                        det_jb_S_np = self.compute_det_1_row(mo_overlap_np, j, b)
                                        det_ia_S_np = self.compute_det_1_row(mo_overlap_np, i, a)
                                        det_jb_S_klcd_np = self.compute_det_1_row_2_column(mo_overlap_np, j, b, k, l, c, d)
                                        det_ia_S_kc_np = self.compute_det_1_row_1_column(mo_overlap_np, i, a, k, c)
                                        det_jb_S_ld_np = self.compute_det_1_row_1_column(mo_overlap_np, j, b, l, d)
                                        det_ia_S_kd_np = self.compute_det_1_row_1_column(mo_overlap_np, i, a, k, d)
                                        det_jb_S_lc_np = self.compute_det_1_row_1_column(mo_overlap_np, j, b, l, c)

                                        # Negative/Negative
                                        S_nn = mo_overlap_nn.copy()
                                        det_S_nn = np.linalg.det(S_nn[0:self.ndocc,0:self.ndocc])
                                        det_ijab_S_klcd_nn = self.compute_det_2_row_2_column(mo_overlap_nn, i, j, a, b, k, l, c, d)
                                        det_ijab_S_nn = self.compute_det_2_row(mo_overlap_nn, i, j, a, b)
                                        det_S_klcd_nn = self.compute_det_2_column(mo_overlap_nn, k, l, c, d)
                                        det_ijab_S_kc_nn = self.compute_det_2_row_1_column(mo_overlap_nn, i, j, a, b, k, c)
                                        det_S_ld_nn = self.compute_det_1_column(mo_overlap_nn, l, d)
                                        det_ijab_S_kd_nn = self.compute_det_2_row_1_column(mo_overlap_nn, i, j, a, b, k, d)
                                        det_S_lc_nn = self.compute_det_1_column(mo_overlap_nn, l, c)

                                        det_ia_S_klcd_nn = self.compute_det_1_row_2_column(mo_overlap_nn, i, a, k, l, c, d)
                                        det_jb_S_nn = self.compute_det_1_row(mo_overlap_nn, j, b)
                                        det_ia_S_nn = self.compute_det_1_row(mo_overlap_nn, i, a)
                                        det_jb_S_klcd_nn = self.compute_det_1_row_2_column(mo_overlap_nn, j, b, k, l, c, d)
                                        det_ia_S_kc_nn = self.compute_det_1_row_1_column(mo_overlap_nn, i, a, k, c)
                                        det_jb_S_ld_nn = self.compute_det_1_row_1_column(mo_overlap_nn, j, b, l, d)
                                        det_ia_S_kd_nn = self.compute_det_1_row_1_column(mo_overlap_nn, i, a, k, d)
                                        det_jb_S_lc_nn = self.compute_det_1_row_1_column(mo_overlap_nn, j, b, l, c)

                                        I += 0.125 * (t_ijab - t_ijba) * (t_klcd - t_kldc) * ((1/(N_nuc_pos*N_mag_pos)) * det_ijab_S_klcd_pp * det_S_pp - (1/(N_nuc_pos*N_mag_neg)) * det_ijab_S_klcd_pn * det_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ijab_S_klcd_np * det_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ijab_S_klcd_nn * det_S_nn)
                                        I += 0.125 * (t_ijab - t_ijba) * (t_klcd - t_kldc) * ((1/(N_nuc_pos*N_mag_pos)) * det_ijab_S_pp * det_S_klcd_pp - (1/(N_nuc_pos*N_mag_neg)) * det_ijab_S_pn * det_S_klcd_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ijab_S_np * det_S_klcd_np + (1/(N_nuc_neg*N_mag_neg)) * det_ijab_S_nn * det_S_klcd_nn)
                                        I += 0.125 * (t_ijab - t_ijba) * 4 * t_klcd * ((1/(N_nuc_pos*N_mag_pos)) * det_ijab_S_kc_pp * det_S_ld_pp - (1/(N_nuc_pos*N_mag_neg)) * det_ijab_S_kc_pn * det_S_ld_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ijab_S_kc_np * det_S_ld_np + (1/(N_nuc_neg*N_mag_neg)) * det_ijab_S_kc_nn * det_S_ld_nn)

                                        I += 2 * 0.125 * t_ijab * (t_klcd - t_kldc) * ((1/(N_nuc_pos*N_mag_pos)) * det_ia_S_klcd_pp * det_jb_S_pp - (1/(N_nuc_pos*N_mag_neg)) * det_ia_S_klcd_pn * det_jb_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ia_S_klcd_np * det_jb_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ia_S_klcd_nn * det_jb_S_nn)
                                        I += 2 * 0.125 * t_ijab * (t_klcd - t_kldc) * ((1/(N_nuc_pos*N_mag_pos)) * det_ia_S_pp * det_jb_S_klcd_pp - (1/(N_nuc_pos*N_mag_neg)) * det_ia_S_pn * det_jb_S_klcd_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ia_S_np * det_jb_S_klcd_np + (1/(N_nuc_neg*N_mag_neg)) * det_ia_S_nn * det_jb_S_klcd_nn)
                                        I += 2 * 0.125 * t_ijab * 4 * t_klcd * ((1/(N_nuc_pos*N_mag_pos)) * det_ia_S_kc_pp * det_jb_S_ld_pp - (1/(N_nuc_pos*N_mag_neg)) * det_ia_S_kc_pn * det_jb_S_ld_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ia_S_kc_np * det_jb_S_ld_np + (1/(N_nuc_neg*N_mag_neg)) * det_ia_S_kc_nn * det_jb_S_ld_nn)



        return I * (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength))




