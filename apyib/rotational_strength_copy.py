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
        S[i,:] = S[a,:]
        det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving a single column swap.
    def compute_det_1_column(self, mo_overlap, k, c):
        S = mo_overlap.copy()
        S[:,k] = S[:,c]
        det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S
    
    # Computes the determinant of the occupied space involving a double row swap.
    def compute_det_2_row(self, mo_overlap, i, j, a, b):
        S = mo_overlap.copy()
        S[i,:] = S[a,:]
        S[j,:] = S[b,:]
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
        S[:,k] = S[:,c]
        S[:,l] = S[:,d]
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
        S[i,:] = S[a,:]
        S[:,k] = S[:,c]
        det_S = np.linalg.det(S[0:self.ndocc,0:self.ndocc])
        
        return det_S

    # Computes the determinant of the occupied space involving two row swaps and a column swap.
    def compute_det_2_row_1_column(self, mo_overlap, i, j, a, b, k, c):
        S = mo_overlap.copy()
        S[i,:] = S[a,:]
        S[j,:] = S[b,:]
        S[:,k] = S[:,c]
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
        S[i,:] = S[a,:]
        S[:,k] = S[:,c]
        S[:,l] = S[:,d]
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
        S[i,:] = S[a,:]
        S[j,:] = S[b,:]
        S[:,k] = S[:,c]
        S[:,l] = S[:,d]
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
        N_unperturbed = 1#np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_t2), self.unperturbed_t2))
        N_nuc_pos = 1#np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_t2[alpha]), self.nuc_pos_t2[alpha]))
        N_nuc_neg = 1#np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_t2[alpha]), self.nuc_neg_t2[alpha]))
        N_mag_pos = 1#np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_t2[beta]), self.mag_pos_t2[beta]))
        N_mag_neg = 1#np.sqrt(1 + 0.25**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_t2[beta]), self.mag_neg_t2[beta]))

        #print(N_nuc_pos - N_nuc_neg, N_mag_pos - N_mag_neg, N_unperturbed)

        # New terms for normalization factor as a variable.
        #hf_ugpg = np.linalg.det(so_overlap_up[0:2*self.ndocc, 0:2*self.ndocc])
        #hf_ugng = np.linalg.det(so_overlap_un[0:2*self.ndocc, 0:2*self.ndocc])
        #hf_pgug = np.linalg.det(so_overlap_pu[0:2*self.ndocc, 0:2*self.ndocc])
        #hf_ngug = np.linalg.det(so_overlap_nu[0:2*self.ndocc, 0:2*self.ndocc])
        #hf_ugug = np.linalg.det(so_overlap_uu[0:2*self.ndocc, 0:2*self.ndocc])

        # Compute the HF term of the CID AAT.
        I = ( (1/(N_nuc_pos*N_mag_pos)) * hf_pgpg - (1/(N_nuc_pos*N_mag_neg)) * hf_pgng - (1/(N_nuc_neg*N_mag_pos)) * hf_ngpg + (1/(N_nuc_neg*N_mag_neg)) * hf_ngng)
        # Compute the HF term with normalization factor as a variable.
        #I = (1/(N_unperturbed**2)) * (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_pgpg - hf_pgng - hf_ngpg + hf_ngng)
        #I -= (1/(N_unperturbed**3)) * (N_nuc_pos - N_nuc_neg) * (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_ugpg - hf_ugng)
        #I -= (1/(N_unperturbed**3)) * (N_mag_pos - N_mag_neg) * (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_pgug - hf_ngug)
        #I += (1/(N_unperturbed**4)) * (N_nuc_pos - N_nuc_neg) * (N_mag_pos - N_mag_neg) * (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_ugug)
        
        I = 0

        # Using spin-orbital formulation for MP2 and CID contribution to the AATs.
        ndocc = 2 * self.ndocc
        nbf = 2 * self.nbf


        # Compute the terms including only one doubly excited determinant in either the bra or ket. 
        for i in range(0, ndocc):
            for j in range(0, ndocc):
                for a in range(ndocc, nbf):
                    for b in range(ndocc, nbf):

                        # Swap the rows for orbital substituion in the bra.
                        # < ijab | d0/dH >
                        so_overlap_egp = so_overlap_up.copy()
                        so_overlap_egp[i,:] = so_overlap_egp[a,:]
                        so_overlap_egp[j,:] = so_overlap_egp[b,:]

                        so_overlap_egn = so_overlap_un.copy()
                        so_overlap_egn[i,:] = so_overlap_egn[a,:]
                        so_overlap_egn[j,:] = so_overlap_egn[b,:]

                        # < dijab/dR | d0/dH >
                        so_overlap_epgp = so_overlap_pp.copy()
                        so_overlap_epgp[i,:] = so_overlap_epgp[a,:]
                        so_overlap_epgp[j,:] = so_overlap_epgp[b,:]

                        so_overlap_epgn = so_overlap_pn.copy()
                        so_overlap_epgn[i,:] = so_overlap_epgn[a,:]
                        so_overlap_epgn[j,:] = so_overlap_epgn[b,:]

                        so_overlap_engp = so_overlap_np.copy()
                        so_overlap_engp[i,:] = so_overlap_engp[a,:]
                        so_overlap_engp[j,:] = so_overlap_engp[b,:]

                        so_overlap_engn = so_overlap_nn.copy()
                        so_overlap_engn[i,:] = so_overlap_engn[a,:]
                        so_overlap_engn[j,:] = so_overlap_engn[b,:]

                        # Swap the columns for orbital substituion in the ket.
                        # < d0/dR | ijab >
                        so_overlap_gpe = so_overlap_pu.copy()
                        so_overlap_gpe[:,i] = so_overlap_gpe[:,a]
                        so_overlap_gpe[:,j] = so_overlap_gpe[:,b]

                        so_overlap_gne = so_overlap_nu.copy()
                        so_overlap_gne[:,i] = so_overlap_gne[:,a]
                        so_overlap_gne[:,j] = so_overlap_gne[:,b]

                        # < d0/dR | dijab/dH >
                        so_overlap_gpep = so_overlap_pp.copy()
                        so_overlap_gpep[:,i] = so_overlap_gpep[:,a]
                        so_overlap_gpep[:,j] = so_overlap_gpep[:,b]

                        so_overlap_gpen = so_overlap_pn.copy()
                        so_overlap_gpen[:,i] = so_overlap_gpen[:,a]
                        so_overlap_gpen[:,j] = so_overlap_gpen[:,b]

                        so_overlap_gnep = so_overlap_np.copy()
                        so_overlap_gnep[:,i] = so_overlap_gnep[:,a]
                        so_overlap_gnep[:,j] = so_overlap_gnep[:,b]

                        so_overlap_gnen = so_overlap_nn.copy()
                        so_overlap_gnen[:,i] = so_overlap_gnen[:,a]
                        so_overlap_gnen[:,j] = so_overlap_gnen[:,b]

                        # Compute determinant overlap.
                        det_overlap_egp = np.linalg.det(so_overlap_egp[0:ndocc, 0:ndocc])
                        det_overlap_egn = np.linalg.det(so_overlap_egn[0:ndocc, 0:ndocc])

                        det_overlap_epgp = np.linalg.det(so_overlap_epgp[0:ndocc, 0:ndocc])
                        det_overlap_epgn = np.linalg.det(so_overlap_epgn[0:ndocc, 0:ndocc])
                        det_overlap_engp = np.linalg.det(so_overlap_engp[0:ndocc, 0:ndocc])
                        det_overlap_engn = np.linalg.det(so_overlap_engn[0:ndocc, 0:ndocc])

                        det_overlap_gpe = np.linalg.det(so_overlap_gpe[0:ndocc, 0:ndocc])
                        det_overlap_gne = np.linalg.det(so_overlap_gne[0:ndocc, 0:ndocc])

                        det_overlap_gpep = np.linalg.det(so_overlap_gpep[0:ndocc, 0:ndocc])
                        det_overlap_gpen = np.linalg.det(so_overlap_gpen[0:ndocc, 0:ndocc])
                        det_overlap_gnep = np.linalg.det(so_overlap_gnep[0:ndocc, 0:ndocc])
                        det_overlap_gnen = np.linalg.det(so_overlap_gnen[0:ndocc, 0:ndocc])

                        # New terms for normalization factor as a variable.
                        #so_overlap_epg = so_overlap_pu.copy()
                        #so_overlap_epg[[i, a],:] = so_overlap_epg[[a, i],:]
                        #so_overlap_epg[[j, b],:] = so_overlap_epg[[b, j],:]
                        #det_overlap_epg = np.linalg.det(so_overlap_epg[0:ndocc, 0:ndocc])

                        #so_overlap_eng = so_overlap_nu.copy()
                        #so_overlap_eng[[i, a],:] = so_overlap_eng[[a, i],:]
                        #so_overlap_eng[[j, b],:] = so_overlap_eng[[b, j],:]
                        #det_overlap_eng = np.linalg.det(so_overlap_eng[0:ndocc, 0:ndocc])

                        #so_overlap_gep = so_overlap_up.copy()
                        #so_overlap_gep[:,[i, a]] = so_overlap_gep[:,[a, i]]
                        #so_overlap_gep[:,[j, b]] = so_overlap_gep[:,[b, j]]
                        #det_overlap_gep = np.linalg.det(so_overlap_gep[0:ndocc, 0:ndocc])

                        #so_overlap_gen = so_overlap_un.copy()
                        #so_overlap_gen[:,[i, a]] = so_overlap_gen[:,[a, i]]
                        #so_overlap_gen[:,[j, b]] = so_overlap_gen[:,[b, j]]
                        #det_overlap_gen = np.linalg.det(so_overlap_gen[0:ndocc, 0:ndocc])

                        # Compute contribution of this component to the AAT.
                        I += 0.25 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * ((1/(N_unperturbed*N_mag_pos)) * det_overlap_egp - (1/(N_unperturbed*N_mag_neg)) * det_overlap_egn)
                        I += 0.25 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * ((1/(N_nuc_pos*N_mag_pos)) * det_overlap_epgp - (1/(N_nuc_pos*N_mag_neg)) * det_overlap_epgn - (1/(N_nuc_neg*N_mag_pos)) * det_overlap_engp + (1/(N_nuc_neg*N_mag_neg)) * det_overlap_engn)
                        I += 0.25 * (self.mag_pos_t2[beta][i][j][a-ndocc][b-ndocc] - self.mag_neg_t2[beta][i][j][a-ndocc][b-ndocc]) * ((1/(N_nuc_pos*N_unperturbed)) * det_overlap_gpe - (1/(N_nuc_neg*N_unperturbed)) * det_overlap_gne)
                        I += 0.25 * self.unperturbed_t2[i][j][a-ndocc][b-ndocc] * ((1/(N_nuc_pos*N_mag_pos)) * det_overlap_gpep - (1/(N_nuc_pos*N_mag_neg)) * det_overlap_gpen - (1/(N_nuc_neg*N_mag_pos)) * det_overlap_gnep + (1/(N_nuc_neg*N_mag_neg)) * det_overlap_gnen)

                        # Compute contribution of this component for normalization factor as a variable.
                        #I += (1/(N_unperturbed**2)) * 0.25 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * (det_overlap_epgp - det_overlap_epgn - det_overlap_engp + det_overlap_engn)
                        #I += (1/(N_unperturbed**2)) * 0.25 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * (det_overlap_egp - det_overlap_egn) # This term = 0.
                        #I -= (1/(N_unperturbed**3)) * (N_nuc_pos - N_nuc_neg) * 0.25 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * (det_overlap_egp - det_overlap_egn)
                        #I -= (1/(N_unperturbed**3)) * (N_mag_pos - N_mag_neg) * 0.25 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * (det_overlap_epg - det_overlap_eng)

                        #I += (1/(N_unperturbed**2)) * 0.25 * self.unperturbed_t2[i][j][a-ndocc][b-ndocc] * (det_overlap_gpep - det_overlap_gpen - det_overlap_gnep + det_overlap_gnen)
                        #I += (1/(N_unperturbed**2)) * 0.25 * (self.mag_pos_t2[beta][i][j][a-ndocc][b-ndocc] - self.mag_neg_t2[beta][i][j][a-ndocc][b-ndocc]) * (det_overlap_gpe - det_overlap_gne)
                        #I -= (1/(N_unperturbed**3)) * (N_nuc_pos - N_nuc_neg) * 0.25 * self.unperturbed_t2[i][j][a-ndocc][b-ndocc] * (det_overlap_gep - det_overlap_gen)
                        #I -= (1/(N_unperturbed**3)) * (N_mag_pos - N_mag_neg) * 0.25 * self.unperturbed_t2[i][j][a-ndocc][b-ndocc] * (det_overlap_gpe - det_overlap_gne)
                        
                        # Compute the terms including the doubly excited determinant in the bra and ket. 
                        #for k in range(0, ndocc):
                        #    for l in range(0, ndocc):
                        #        for c in range(ndocc, nbf):
                        #            for d in range(ndocc, nbf):

                        #                # Swap the rows and columns for orbital substituion in the bra and ket.
                        #                # < ijab | klcd >
                        #                so_overlap_ee = so_overlap_uu.copy()
                        #                so_overlap_ee[i,:] = so_overlap_ee[a,:]
                        #                so_overlap_ee[j,:] = so_overlap_ee[b,:]
                        #                so_overlap_ee[:,k] = so_overlap_ee[:,c]
                        #                so_overlap_ee[:,l] = so_overlap_ee[:,d]

                        #                # < ijab | dklcd/dH >
                        #                so_overlap_eep = so_overlap_up.copy()
                        #                so_overlap_eep[i,:] = so_overlap_eep[a,:]
                        #                so_overlap_eep[j,:] = so_overlap_eep[b,:]
                        #                so_overlap_eep[:,k] = so_overlap_eep[:,c]
                        #                so_overlap_eep[:,l] = so_overlap_eep[:,d]

                        #                so_overlap_een = so_overlap_un.copy()
                        #                so_overlap_een[i,:] = so_overlap_een[a,:]
                        #                so_overlap_een[j,:] = so_overlap_een[b,:]
                        #                so_overlap_een[:,k] = so_overlap_een[:,c]
                        #                so_overlap_een[:,l] = so_overlap_een[:,d]

                        #                # < dijab/dR | klcd >
                        #                so_overlap_epe = so_overlap_pu.copy()
                        #                so_overlap_epe[i,:] = so_overlap_epe[a,:]
                        #                so_overlap_epe[j,:] = so_overlap_epe[b,:]
                        #                so_overlap_epe[:,k] = so_overlap_epe[:,c]
                        #                so_overlap_epe[:,l] = so_overlap_epe[:,d]

                        #                so_overlap_ene = so_overlap_nu.copy()
                        #                so_overlap_ene[i,:] = so_overlap_ene[a,:]
                        #                so_overlap_ene[j,:] = so_overlap_ene[b,:]
                        #                so_overlap_ene[:,k] = so_overlap_ene[:,c]
                        #                so_overlap_ene[:,l] = so_overlap_ene[:,d]

                        #                # < dijab/dR | dklcd/dH >
                        #                so_overlap_epep = so_overlap_pp.copy()
                        #                so_overlap_epep[i,:] = so_overlap_epep[a,:]
                        #                so_overlap_epep[j,:] = so_overlap_epep[b,:]
                        #                so_overlap_epep[:,k] = so_overlap_epep[:,c]
                        #                so_overlap_epep[:,l] = so_overlap_epep[:,d]

                        #                so_overlap_epen = so_overlap_pn.copy()
                        #                so_overlap_epen[i,:] = so_overlap_epen[a,:]
                        #                so_overlap_epen[j,:] = so_overlap_epen[b,:]
                        #                so_overlap_epen[:,k] = so_overlap_epen[:,c]
                        #                so_overlap_epen[:,l] = so_overlap_epen[:,d]

                        #                so_overlap_enep = so_overlap_np.copy()
                        #                so_overlap_enep[i,:] = so_overlap_enep[a,:]
                        #                so_overlap_enep[j,:] = so_overlap_enep[b,:]
                        #                so_overlap_enep[:,k] = so_overlap_enep[:,c]
                        #                so_overlap_enep[:,l] = so_overlap_enep[:,d]

                        #                so_overlap_enen = so_overlap_nn.copy()
                        #                so_overlap_enen[i,:] = so_overlap_enen[a,:]
                        #                so_overlap_enen[j,:] = so_overlap_enen[b,:]
                        #                so_overlap_enen[:,k] = so_overlap_enen[:,c]
                        #                so_overlap_enen[:,l] = so_overlap_enen[:,d]

                        #                # Compute determinant overlap.
                        #                det_overlap_ee = np.linalg.det(so_overlap_ee[0:ndocc, 0:ndocc])

                        #                det_overlap_eep = np.linalg.det(so_overlap_eep[0:ndocc, 0:ndocc])
                        #                det_overlap_een = np.linalg.det(so_overlap_een[0:ndocc, 0:ndocc])

                        #                det_overlap_epe = np.linalg.det(so_overlap_epe[0:ndocc, 0:ndocc])
                        #                det_overlap_ene = np.linalg.det(so_overlap_ene[0:ndocc, 0:ndocc])

                        #                det_overlap_epep = np.linalg.det(so_overlap_epep[0:ndocc, 0:ndocc])
                        #                det_overlap_epen = np.linalg.det(so_overlap_epen[0:ndocc, 0:ndocc])
                        #                det_overlap_enep = np.linalg.det(so_overlap_enep[0:ndocc, 0:ndocc])
                        #                det_overlap_enen = np.linalg.det(so_overlap_enen[0:ndocc, 0:ndocc])

                        #                # Contribution to this component was not computed for the normalization factor as a variable case.

                        #                # Compute contribution of this component to the AAT.
                        #                I += 0.25**2 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * (self.mag_pos_t2[beta][k][l][c-ndocc][d-ndocc] - self.mag_neg_t2[beta][k][l][c-ndocc][d-ndocc]) * (1/(N_unperturbed**2)) * det_overlap_ee
                        #                #I += 0.25**2 * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-ndocc][b-ndocc] - self.nuc_neg_t2[alpha][i][j][a-ndocc][b-ndocc])) * self.unperturbed_t2[k][l][c-ndocc][d-ndocc] * ((1/(N_unperturbed*N_mag_pos)) * det_overlap_eep - (1/(N_unperturbed*N_mag_neg)) * det_overlap_een)
                        #                #I += 0.25**2 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * (self.mag_pos_t2[beta][k][l][c-ndocc][d-ndocc] - self.mag_neg_t2[beta][k][l][c-ndocc][d-ndocc]) * ((1/(N_nuc_pos*N_unperturbed)) * det_overlap_epe - (1/(N_nuc_neg*N_unperturbed)) * det_overlap_ene)
                        #                #I += 0.25**2 * np.conjugate(self.unperturbed_t2[i][j][a-ndocc][b-ndocc]) * self.unperturbed_t2[k][l][c-ndocc][d-ndocc] * ((1/(N_nuc_pos*N_mag_pos)) * det_overlap_epep - (1/(N_nuc_pos*N_mag_neg)) * det_overlap_epen - (1/(N_nuc_neg*N_mag_pos)) * det_overlap_enep + (1/(N_nuc_neg*N_mag_neg)) * det_overlap_enen)

 
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
        N_unperturbed = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_t2), self.unperturbed_t2))
        N_nuc_pos = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_t2[alpha]), self.nuc_pos_t2[alpha]))
        N_nuc_neg = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_t2[alpha]), self.nuc_neg_t2[alpha]))
        N_mag_pos = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_t2[beta]), self.mag_pos_t2[beta]))
        N_mag_neg = 1#np.sqrt(1 + 0.5**2 * np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_t2[beta]), self.mag_neg_t2[beta]))

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

                        ijab_S_up = mo_overlap_up.copy()
                        ijab_S_up[i,:] = ijab_S_up[a,:]
                        ijab_S_up[j,:] = ijab_S_up[b,:]
                        det_ijab_S_up = np.linalg.det(ijab_S_up[0:self.ndocc,0:self.ndocc])

                        ia_S_up = mo_overlap_up.copy()
                        ia_S_up[i,:] = ia_S_up[a,:]
                        det_ia_S_up = np.linalg.det(ia_S_up[0:self.ndocc,0:self.ndocc])

                        jb_S_up = mo_overlap_up.copy()
                        jb_S_up[j,:] = jb_S_up[b,:]
                        det_jb_S_up = np.linalg.det(jb_S_up[0:self.ndocc,0:self.ndocc])

                        ib_S_up = mo_overlap_up.copy()
                        ib_S_up[i,:] = ib_S_up[b,:]
                        det_ib_S_up = np.linalg.det(ib_S_up[0:self.ndocc,0:self.ndocc])

                        ja_S_up = mo_overlap_up.copy()
                        ja_S_up[j,:] = ja_S_up[a,:]
                        det_ja_S_up = np.linalg.det(ja_S_up[0:self.ndocc,0:self.ndocc])

                        t_ijab_p = self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_p = self.nuc_pos_t2[alpha][i][j][b-self.ndocc][a-self.ndocc]

                        # Unperturbed/Negative
                        S_un = mo_overlap_un.copy()
                        det_S_un = np.linalg.det(S_un[0:self.ndocc,0:self.ndocc])

                        ijab_S_un = mo_overlap_un.copy()
                        ijab_S_un[i,:] = ijab_S_un[a,:]
                        ijab_S_un[j,:] = ijab_S_un[b,:]
                        det_ijab_S_un = np.linalg.det(ijab_S_un[0:self.ndocc,0:self.ndocc])

                        ia_S_un = mo_overlap_un.copy()
                        ia_S_un[i,:] = ia_S_un[a,:]
                        det_ia_S_un = np.linalg.det(ia_S_un[0:self.ndocc,0:self.ndocc])

                        jb_S_un = mo_overlap_un.copy()
                        jb_S_un[j,:] = jb_S_un[b,:]
                        det_jb_S_un = np.linalg.det(jb_S_un[0:self.ndocc,0:self.ndocc])

                        ib_S_un = mo_overlap_un.copy()
                        ib_S_un[i,:] = ib_S_un[b,:]
                        det_ib_S_un = np.linalg.det(ib_S_un[0:self.ndocc,0:self.ndocc])

                        ja_S_un = mo_overlap_un.copy()
                        ja_S_un[j,:] = ja_S_un[a,:]
                        det_ja_S_un = np.linalg.det(ja_S_un[0:self.ndocc,0:self.ndocc])

                        t_ijab_n = self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_n = self.nuc_neg_t2[alpha][i][j][b-self.ndocc][a-self.ndocc]

                        #I += 0.5 * np.conjugate((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * ((1/(N_unperturbed*N_mag_pos)) * det_ijab_S_up * det_S_up  - (1/(N_unperturbed*N_mag_neg)) * det_ijab_S_un * det_S_un)
                        #I += 0.5 * np.conjugate((t_ijab_p - t_ijab_n)) * ((1/(N_unperturbed*N_mag_pos)) * det_ia_S_up * det_jb_S_up  - (1/(N_unperturbed*N_mag_neg)) * det_ia_S_un * det_jb_S_un)
                        #I += 0.5 * np.conjugate((t_ijba_p - t_ijba_n)) * ((1/(N_unperturbed*N_mag_pos)) * det_ib_S_up * det_ja_S_up  - (1/(N_unperturbed*N_mag_neg)) * det_ib_S_un * det_ja_S_un)

                        # < dijab/dR | d0/dH >
                        # Positive/Positive
                        S_pp = mo_overlap_pp.copy()
                        det_S_pp = np.linalg.det(S_pp[0:self.ndocc,0:self.ndocc])

                        ijab_S_pp = mo_overlap_pp.copy()
                        ijab_S_pp[i,:] = ijab_S_pp[a,:]
                        ijab_S_pp[j,:] = ijab_S_pp[b,:]
                        det_ijab_S_pp = np.linalg.det(ijab_S_pp[0:self.ndocc,0:self.ndocc])

                        ia_S_pp = mo_overlap_pp.copy()
                        ia_S_pp[i,:] = ia_S_pp[a,:]
                        det_ia_S_pp = np.linalg.det(ia_S_pp[0:self.ndocc,0:self.ndocc])

                        jb_S_pp = mo_overlap_pp.copy()
                        jb_S_pp[j,:] = jb_S_pp[b,:]
                        det_jb_S_pp = np.linalg.det(jb_S_pp[0:self.ndocc,0:self.ndocc])

                        ib_S_pp = mo_overlap_pp.copy()
                        ib_S_pp[i,:] = ib_S_pp[b,:]
                        det_ib_S_pp = np.linalg.det(ib_S_pp[0:self.ndocc,0:self.ndocc])

                        ja_S_pp = mo_overlap_pp.copy()
                        ja_S_pp[j,:] = ja_S_pp[a,:]
                        det_ja_S_pp = np.linalg.det(ja_S_pp[0:self.ndocc,0:self.ndocc])

                        # Negative/Negative
                        S_nn = mo_overlap_nn.copy()
                        det_S_nn = np.linalg.det(S_nn[0:self.ndocc,0:self.ndocc])

                        ijab_S_nn = mo_overlap_nn.copy()
                        ijab_S_nn[i,:] = ijab_S_nn[a,:]
                        ijab_S_nn[j,:] = ijab_S_nn[b,:]
                        det_ijab_S_nn = np.linalg.det(ijab_S_nn[0:self.ndocc,0:self.ndocc])

                        ia_S_nn = mo_overlap_nn.copy()
                        ia_S_nn[i,:] = ia_S_nn[a,:]
                        det_ia_S_nn = np.linalg.det(ia_S_nn[0:self.ndocc,0:self.ndocc])

                        jb_S_nn = mo_overlap_nn.copy()
                        jb_S_nn[j,:] = jb_S_nn[b,:]
                        det_jb_S_nn = np.linalg.det(jb_S_nn[0:self.ndocc,0:self.ndocc])

                        ib_S_nn = mo_overlap_nn.copy()
                        ib_S_nn[i,:] = ib_S_nn[b,:]
                        det_ib_S_nn = np.linalg.det(ib_S_nn[0:self.ndocc,0:self.ndocc])

                        ja_S_nn = mo_overlap_nn.copy()
                        ja_S_nn[j,:] = ja_S_nn[a,:]
                        det_ja_S_nn = np.linalg.det(ja_S_nn[0:self.ndocc,0:self.ndocc])

                        # Positive/Negative
                        S_pn = mo_overlap_pn.copy()
                        det_S_pn = np.linalg.det(S_pn[0:self.ndocc,0:self.ndocc])

                        ijab_S_pn = mo_overlap_pn.copy()
                        ijab_S_pn[i,:] = ijab_S_pn[a,:]
                        ijab_S_pn[j,:] = ijab_S_pn[b,:]
                        det_ijab_S_pn = np.linalg.det(ijab_S_pn[0:self.ndocc,0:self.ndocc])

                        ia_S_pn = mo_overlap_pn.copy()
                        ia_S_pn[i,:] = ia_S_pn[a,:]
                        det_ia_S_pn = np.linalg.det(ia_S_pn[0:self.ndocc,0:self.ndocc])

                        jb_S_pn = mo_overlap_pn.copy()
                        jb_S_pn[j,:] = jb_S_pn[b,:]
                        det_jb_S_pn = np.linalg.det(jb_S_pn[0:self.ndocc,0:self.ndocc])

                        ib_S_pn = mo_overlap_pn.copy()
                        ib_S_pn[i,:] = ib_S_pn[b,:]
                        det_ib_S_pn = np.linalg.det(ib_S_pn[0:self.ndocc,0:self.ndocc])

                        ja_S_pn = mo_overlap_pn.copy()
                        ja_S_pn[j,:] = ja_S_pn[a,:]
                        det_ja_S_pn = np.linalg.det(ja_S_pn[0:self.ndocc,0:self.ndocc])

                        # Negative/Positive
                        S_np = mo_overlap_np.copy()
                        det_S_np = np.linalg.det(S_np[0:self.ndocc,0:self.ndocc])

                        ijab_S_np = mo_overlap_np.copy()
                        ijab_S_np[i,:] = ijab_S_np[a,:]
                        ijab_S_np[j,:] = ijab_S_np[b,:]
                        det_ijab_S_np = np.linalg.det(ijab_S_np[0:self.ndocc,0:self.ndocc])

                        ia_S_np = mo_overlap_np.copy()
                        ia_S_np[i,:] = ia_S_np[a,:]
                        det_ia_S_np = np.linalg.det(ia_S_np[0:self.ndocc,0:self.ndocc])

                        jb_S_np = mo_overlap_np.copy()
                        jb_S_np[j,:] = jb_S_np[b,:]
                        det_jb_S_np = np.linalg.det(jb_S_np[0:self.ndocc,0:self.ndocc])

                        ib_S_np = mo_overlap_np.copy()
                        ib_S_np[i,:] = ib_S_np[b,:]
                        det_ib_S_np = np.linalg.det(ib_S_np[0:self.ndocc,0:self.ndocc])

                        ja_S_np = mo_overlap_np.copy()
                        ja_S_np[j,:] = ja_S_np[a,:]
                        det_ja_S_np = np.linalg.det(ja_S_np[0:self.ndocc,0:self.ndocc])

                        # Amplitudes
                        t_ijab = self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba = self.unperturbed_t2[i][j][b-self.ndocc][a-self.ndocc]

                        #I += 0.5 * np.conjugate((t_ijab - t_ijba)) * ((1/(N_nuc_pos*N_mag_pos)) * det_ijab_S_pp * det_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_ijab_S_pn * det_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ijab_S_np * det_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ijab_S_nn * det_S_nn)
                        #I += 0.5 * np.conjugate(t_ijab) * ((1/(N_nuc_pos*N_mag_pos)) * det_ia_S_pp * det_jb_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_ia_S_pn * det_jb_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ia_S_np * det_jb_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ia_S_nn * det_jb_S_nn)
                        #I += 0.5 * np.conjugate(t_ijba) * ((1/(N_nuc_pos*N_mag_pos)) * det_ib_S_pp * det_ja_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_ib_S_pn * det_ja_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_ib_S_np * det_ja_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_ib_S_nn * det_ja_S_nn)

                        # Swap the columns for orbital substituion in the ket.
                        # < d0/dR | ijab >
                        # Positive
                        S_pu = mo_overlap_pu.copy()
                        det_S_pu = np.linalg.det(S_pu[0:self.ndocc,0:self.ndocc])

                        S_ijab_pu = mo_overlap_pu.copy()
                        S_ijab_pu[:,i] = S_ijab_pu[:,a]
                        S_ijab_pu[:,j] = S_ijab_pu[:,b]
                        det_S_ijab_pu = np.linalg.det(S_ijab_pu[0:self.ndocc,0:self.ndocc])

                        S_ia_pu = mo_overlap_pu.copy()
                        S_ia_pu[:,i] = S_ia_pu[:,a]
                        det_S_ia_pu = np.linalg.det(S_ia_pu[0:self.ndocc,0:self.ndocc])

                        S_jb_pu = mo_overlap_pu.copy()
                        S_jb_pu[:,j] = S_jb_pu[:,b]
                        det_S_jb_pu = np.linalg.det(S_jb_pu[0:self.ndocc,0:self.ndocc])

                        S_ib_pu = mo_overlap_pu.copy()
                        S_ib_pu[:,i] = S_ib_pu[:,b]
                        det_S_ib_pu = np.linalg.det(S_ib_pu[0:self.ndocc,0:self.ndocc])

                        S_ja_pu = mo_overlap_pu.copy()
                        S_ja_pu[:,j] = S_ja_pu[:,a]
                        det_S_ja_pu = np.linalg.det(S_ja_pu[0:self.ndocc,0:self.ndocc])

                        t_ijab_p = self.mag_pos_t2[beta][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_p = self.mag_pos_t2[beta][i][j][b-self.ndocc][a-self.ndocc]

                        # Negative
                        S_nu = mo_overlap_nu.copy()
                        det_S_nu = np.linalg.det(S_nu[0:self.ndocc,0:self.ndocc])

                        S_ijab_nu = mo_overlap_nu.copy()
                        S_ijab_nu[:,i] = S_ijab_nu[:,a]
                        S_ijab_nu[:,j] = S_ijab_nu[:,b]
                        det_S_ijab_nu = np.linalg.det(S_ijab_nu[0:self.ndocc,0:self.ndocc])

                        S_ia_nu = mo_overlap_nu.copy()
                        S_ia_nu[:,i] = S_ia_nu[:,a]
                        det_S_ia_nu = np.linalg.det(S_ia_nu[0:self.ndocc,0:self.ndocc])

                        S_jb_nu = mo_overlap_nu.copy()
                        S_jb_nu[:,j] = S_jb_nu[:,b]
                        det_S_jb_nu = np.linalg.det(S_jb_nu[0:self.ndocc,0:self.ndocc])

                        S_ib_nu = mo_overlap_nu.copy()
                        S_ib_nu[:,i] = S_ib_nu[:,b]
                        det_S_ib_nu = np.linalg.det(S_ib_nu[0:self.ndocc,0:self.ndocc])

                        S_ja_nu = mo_overlap_nu.copy()
                        S_ja_nu[:,j] = S_ja_nu[:,a]
                        det_S_ja_nu = np.linalg.det(S_ja_nu[0:self.ndocc,0:self.ndocc])

                        t_ijab_n = self.mag_neg_t2[beta][i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba_n = self.mag_neg_t2[beta][i][j][b-self.ndocc][a-self.ndocc]

                        #I += 0.5 * ((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * ((1/(N_unperturbed*N_nuc_pos)) * det_S_ijab_pu * det_S_pu  - (1/(N_unperturbed*N_nuc_neg)) * det_S_ijab_nu * det_S_nu)
                        #I += 0.5 * (t_ijab_p - t_ijab_n) * ((1/(N_unperturbed*N_nuc_pos)) * det_S_ia_pu * det_S_jb_pu  - (1/(N_unperturbed*N_nuc_neg)) * det_S_ia_nu * det_S_jb_nu)
                        #I += 0.5 * (t_ijba_p - t_ijba_n) * ((1/(N_unperturbed*N_nuc_pos)) * det_S_ib_pu * det_S_ja_pu  - (1/(N_unperturbed*N_nuc_neg)) * det_S_ib_nu * det_S_ja_nu)

                        # < d0/dR | dijab/dH >
                        # Positive/Positive
                        S_pp = mo_overlap_pp.copy()
                        det_S_pp = np.linalg.det(S_pp[0:self.ndocc,0:self.ndocc])

                        S_ijab_pp = mo_overlap_pp.copy()
                        S_ijab_pp[:,i] = S_ijab_pp[:,a]
                        S_ijab_pp[:,j] = S_ijab_pp[:,b]
                        det_S_ijab_pp = np.linalg.det(S_ijab_pp[0:self.ndocc,0:self.ndocc])

                        S_ia_pp = mo_overlap_pp.copy()
                        S_ia_pp[:,i] = S_ia_pp[:,a]
                        det_S_ia_pp = np.linalg.det(S_ia_pp[0:self.ndocc,0:self.ndocc])

                        S_jb_pp = mo_overlap_pp.copy()
                        S_jb_pp[:,j] = S_jb_pp[:,b]
                        det_S_jb_pp = np.linalg.det(S_jb_pp[0:self.ndocc,0:self.ndocc])

                        S_ib_pp = mo_overlap_pp.copy()
                        S_ib_pp[:,i] = S_ib_pp[:,b]
                        det_S_ib_pp = np.linalg.det(S_ib_pp[0:self.ndocc,0:self.ndocc])

                        S_ja_pp = mo_overlap_pp.copy()
                        S_ja_pp[:,j] = S_ja_pp[:,a]
                        det_S_ja_pp = np.linalg.det(S_ja_pp[0:self.ndocc,0:self.ndocc])

                        # Negative/Negative
                        S_nn = mo_overlap_nn.copy()
                        det_S_nn = np.linalg.det(S_nn[0:self.ndocc,0:self.ndocc])

                        S_ijab_nn = mo_overlap_nn.copy()
                        S_ijab_nn[:,i] = S_ijab_nn[:,a]
                        S_ijab_nn[:,j] = S_ijab_nn[:,b]
                        det_S_ijab_nn = np.linalg.det(S_ijab_nn[0:self.ndocc,0:self.ndocc])

                        S_ia_nn = mo_overlap_nn.copy()
                        S_ia_nn[:,i] = S_ia_nn[:,a]
                        det_S_ia_nn = np.linalg.det(S_ia_nn[0:self.ndocc,0:self.ndocc])

                        S_jb_nn = mo_overlap_nn.copy()
                        S_jb_nn[:,j] = S_jb_nn[:,b]
                        det_S_jb_nn = np.linalg.det(S_jb_nn[0:self.ndocc,0:self.ndocc])

                        S_ib_nn = mo_overlap_nn.copy()
                        S_ib_nn[:,i] = S_ib_nn[:,b]
                        det_S_ib_nn = np.linalg.det(S_ib_nn[0:self.ndocc,0:self.ndocc])

                        S_ja_nn = mo_overlap_nn.copy()
                        S_ja_nn[:,j] = S_ja_nn[:,a]
                        det_S_ja_nn = np.linalg.det(S_ja_nn[0:self.ndocc,0:self.ndocc])

                        # Positive/Negative
                        S_pn = mo_overlap_pn.copy()
                        det_S_pn = np.linalg.det(S_pn[0:self.ndocc,0:self.ndocc])

                        S_ijab_pn = mo_overlap_pn.copy()
                        S_ijab_pn[:,i] = S_ijab_pn[:,a]
                        S_ijab_pn[:,j] = S_ijab_pn[:,b]
                        det_S_ijab_pn = np.linalg.det(S_ijab_pn[0:self.ndocc,0:self.ndocc])

                        S_ia_pn = mo_overlap_pn.copy()
                        S_ia_pn[:,i] = S_ia_pn[:,a]
                        det_S_ia_pn = np.linalg.det(S_ia_pn[0:self.ndocc,0:self.ndocc])

                        S_jb_pn = mo_overlap_pn.copy()
                        S_jb_pn[:,j] = S_jb_pn[:,b]
                        det_S_jb_pn = np.linalg.det(S_jb_pn[0:self.ndocc,0:self.ndocc])

                        S_ib_pn = mo_overlap_pn.copy()
                        S_ib_pn[:,i] = S_ib_pn[:,b]
                        det_S_ib_pn = np.linalg.det(S_ib_pn[0:self.ndocc,0:self.ndocc])

                        S_ja_pn = mo_overlap_pn.copy()
                        S_ja_pn[:,j] = S_ja_pn[:,a]
                        det_S_ja_pn = np.linalg.det(S_ja_pn[0:self.ndocc,0:self.ndocc])

                        # Negative/Positive
                        S_np = mo_overlap_np.copy()
                        det_S_np = np.linalg.det(S_np[0:self.ndocc,0:self.ndocc])

                        S_ijab_np = mo_overlap_np.copy()
                        S_ijab_np[:,i] = S_ijab_np[:,a]
                        S_ijab_np[:,j] = S_ijab_np[:,b]
                        det_S_ijab_np = np.linalg.det(S_ijab_np[0:self.ndocc,0:self.ndocc])

                        S_ia_np = mo_overlap_np.copy()
                        S_ia_np[:,i] = S_ia_np[:,a]
                        det_S_ia_np = np.linalg.det(S_ia_np[0:self.ndocc,0:self.ndocc])

                        S_jb_np = mo_overlap_np.copy()
                        S_jb_np[:,j] = S_jb_np[:,b]
                        det_S_jb_np = np.linalg.det(S_jb_np[0:self.ndocc,0:self.ndocc])

                        S_ib_np = mo_overlap_np.copy()
                        S_ib_np[:,i] = S_ib_np[:,b]
                        det_S_ib_np = np.linalg.det(S_ib_np[0:self.ndocc,0:self.ndocc])

                        S_ja_np = mo_overlap_np.copy()
                        S_ja_np[:,j] = S_ja_np[:,a]
                        det_S_ja_np = np.linalg.det(S_ja_np[0:self.ndocc,0:self.ndocc])

                        # Amplitudes
                        t_ijab = self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]
                        t_ijba = self.unperturbed_t2[i][j][b-self.ndocc][a-self.ndocc]

                        #I += 0.5 * (t_ijab - t_ijba) * ((1/(N_nuc_pos*N_mag_pos)) * det_S_ijab_pp * det_S_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_S_ijab_pn * det_S_pn - (1/(N_nuc_neg*N_mag_pos)) * det_S_ijab_np * det_S_np + (1/(N_nuc_neg*N_mag_neg)) * det_S_ijab_nn * det_S_nn)
                        #I += 0.5 * (t_ijab) * ((1/(N_nuc_pos*N_mag_pos)) * det_S_ia_pp * det_S_jb_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_S_ia_pn * det_S_jb_pn - (1/(N_nuc_neg*N_mag_pos)) * det_S_ia_np * det_S_jb_np + (1/(N_nuc_neg*N_mag_neg)) * det_S_ia_nn * det_S_jb_nn)
                        #I += 0.5 * (t_ijba) * ((1/(N_nuc_pos*N_mag_pos)) * det_S_ib_pp * det_S_ja_pp  - (1/(N_nuc_pos*N_mag_neg)) * det_S_ib_pn * det_S_ja_pn - (1/(N_nuc_neg*N_mag_pos)) * det_S_ib_np * det_S_ja_np + (1/(N_nuc_neg*N_mag_neg)) * det_S_ib_nn * det_S_ja_nn)

                        for k in range(0, self.ndocc):
                            for l in range(0, self.ndocc):
                                for c in range(self.ndocc, self.nbf):
                                    for d in range(self.ndocc, self.nbf):

                                        # Swap the rows for orbital substituion in the bra.
                                        # < ijab | klcd >
                                        # Unperturbed
                                        S_uu = mo_overlap_uu.copy()
                                        det_S_uu = np.linalg.det(S_uu[0:self.ndocc,0:self.ndocc])

                                        ijab_S_klcd_uu = mo_overlap_uu.copy()
                                        ijab_S_klcd_uu[i,:] = ijab_S_klcd_uu[a,:]
                                        ijab_S_klcd_uu[j,:] = ijab_S_klcd_uu[b,:]
                                        ijab_S_klcd_uu[:,k] = ijab_S_klcd_uu[:,c]
                                        ijab_S_klcd_uu[:,l] = ijab_S_klcd_uu[:,d]
                                        det_ijab_S_klcd_uu = np.linalg.det(ijab_S_klcd_uu[0:self.ndocc,0:self.ndocc])

                                        ijab_S_uu = mo_overlap_uu.copy()
                                        ijab_S_uu[i,:] = ijab_S_uu[a,:]
                                        ijab_S_uu[j,:] = ijab_S_uu[b,:]
                                        det_ijab_S_uu = np.linalg.det(ijab_S_uu[0:self.ndocc,0:self.ndocc])                                        

                                        S_klcd_uu = mo_overlap_uu.copy()
                                        S_klcd_uu[:,k] = S_klcd_uu[:,c]
                                        S_klcd_uu[:,l] = S_klcd_uu[:,d]
                                        det_S_klcd_uu = np.linalg.det(S_klcd_uu[0:self.ndocc,0:self.ndocc])

                                        ijab_S_kc_uu = mo_overlap_uu.copy()
                                        ijab_S_kc_uu[i,:] = ijab_S_kc_uu[a,:]
                                        ijab_S_kc_uu[j,:] = ijab_S_kc_uu[b,:]
                                        ijab_S_kc_uu[:,k] = ijab_S_kc_uu[:,c]
                                        det_ijab_S_kc_uu = np.linalg.det(ijab_S_kc_uu[0:self.ndocc,0:self.ndocc])

                                        S_ld_uu = mo_overlap_uu.copy()
                                        S_ld_uu[:,l] = S_ld_uu[:,d]
                                        det_S_ld_uu = np.linalg.det(S_ld_uu[0:self.ndocc,0:self.ndocc])

                                        ijab_S_kd_uu = mo_overlap_uu.copy()
                                        ijab_S_kd_uu[i,:] = ijab_S_kd_uu[a,:]
                                        ijab_S_kd_uu[j,:] = ijab_S_kd_uu[b,:]
                                        ijab_S_kd_uu[:,k] = ijab_S_kd_uu[:,d]
                                        det_ijab_S_kd_uu = np.linalg.det(ijab_S_kd_uu[0:self.ndocc,0:self.ndocc])

                                        S_lc_uu = mo_overlap_uu.copy()
                                        S_lc_uu[:,l] = S_lc_uu[:,c]
                                        det_S_lc_uu = np.linalg.det(S_lc_uu[0:self.ndocc,0:self.ndocc])

                                        ###
                                        ia_S_klcd_uu = mo_overlap_uu.copy()
                                        ia_S_klcd_uu[i,:] = ia_S_klcd_uu[a,:]
                                        ia_S_klcd_uu[:,k] = ia_S_klcd_uu[:,c]
                                        ia_S_klcd_uu[:,l] = ia_S_klcd_uu[:,d]
                                        det_ia_S_klcd_uu = np.linalg.det(ia_S_klcd_uu[0:self.ndocc,0:self.ndocc])

                                        jb_S_uu = mo_overlap_uu.copy()
                                        jb_S_uu[j,:] = jb_S_uu[b,:]
                                        det_jb_S_uu = np.linalg.det(jb_S_uu[0:self.ndocc,0:self.ndocc])

                                        ia_S_uu = mo_overlap_uu.copy()
                                        ia_S_uu[i,:] = ia_S_uu[a,:]
                                        det_ia_S_uu = np.linalg.det(ia_S_uu[0:self.ndocc,0:self.ndocc])

                                        jb_S_klcd_uu = mo_overlap_uu.copy()
                                        jb_S_klcd_uu[j,:] = jb_S_klcd_uu[b,:]
                                        jb_S_klcd_uu[:,k] = jb_S_klcd_uu[:,c]
                                        jb_S_klcd_uu[:,l] = jb_S_klcd_uu[:,d]
                                        det_jb_S_klcd_uu = np.linalg.det(jb_S_klcd_uu[0:self.ndocc,0:self.ndocc])

                                        ia_S_kc_uu = mo_overlap_uu.copy()
                                        ia_S_kc_uu[i,:] = ia_S_kc_uu[a,:]
                                        ia_S_kc_uu[:,k] = ia_S_kc_uu[:,c]
                                        det_ia_S_kc_uu = np.linalg.det(ia_S_kc_uu[0:self.ndocc,0:self.ndocc])

                                        jb_S_ld_uu = mo_overlap_uu.copy()
                                        jb_S_ld_uu[j,:] = jb_S_ld_uu[b,:]
                                        jb_S_ld_uu[:,l] = jb_S_ld_uu[:,d]
                                        det_jb_S_ld_uu = np.linalg.det(jb_S_ld_uu[0:self.ndocc,0:self.ndocc])

                                        ia_S_kd_uu = mo_overlap_uu.copy()
                                        ia_S_kd_uu[i,:] = ia_S_kd_uu[a,:]
                                        ia_S_kd_uu[:,k] = ia_S_kd_uu[:,d]
                                        det_ia_S_kd_uu = np.linalg.det(ia_S_kd_uu[0:self.ndocc,0:self.ndocc])

                                        jb_S_lc_uu = mo_overlap_uu.copy()
                                        jb_S_lc_uu[j,:] = jb_S_lc_uu[b,:]
                                        jb_S_lc_uu[:,l] = jb_S_lc_uu[:,c]
                                        det_jb_S_lc_uu = np.linalg.det(jb_S_lc_uu[0:self.ndocc,0:self.ndocc])

                                        ###
                                        ib_S_klcd_uu = mo_overlap_uu.copy()
                                        ib_S_klcd_uu[i,:] = ib_S_klcd_uu[b,:]
                                        ib_S_klcd_uu[:,k] = ib_S_klcd_uu[:,c]
                                        ib_S_klcd_uu[:,l] = ib_S_klcd_uu[:,d]
                                        det_ib_S_klcd_uu = np.linalg.det(ib_S_klcd_uu[0:self.ndocc,0:self.ndocc])

                                        ja_S_uu = mo_overlap_uu.copy()
                                        ja_S_uu[j,:] = ja_S_uu[a,:]
                                        det_ja_S_uu = np.linalg.det(ja_S_uu[0:self.ndocc,0:self.ndocc])

                                        ib_S_uu = mo_overlap_uu.copy()
                                        ib_S_uu[i,:] = ib_S_uu[b,:]
                                        det_ib_S_uu = np.linalg.det(ib_S_uu[0:self.ndocc,0:self.ndocc])

                                        ja_S_klcd_uu = mo_overlap_uu.copy()
                                        ja_S_klcd_uu[j,:] = ja_S_klcd_uu[a,:]
                                        ja_S_klcd_uu[:,k] = ja_S_klcd_uu[:,c]
                                        ja_S_klcd_uu[:,l] = ja_S_klcd_uu[:,d]
                                        det_ja_S_klcd_uu = np.linalg.det(ja_S_klcd_uu[0:self.ndocc,0:self.ndocc])

                                        ib_S_kc_uu = mo_overlap_uu.copy()
                                        ib_S_kc_uu[i,:] = ib_S_kc_uu[b,:]
                                        ib_S_kc_uu[:,k] = ib_S_kc_uu[:,c]
                                        det_ib_S_kc_uu = np.linalg.det(ib_S_kc_uu[0:self.ndocc,0:self.ndocc])

                                        ja_S_ld_uu = mo_overlap_uu.copy()
                                        ja_S_ld_uu[j,:] = ja_S_ld_uu[a,:]
                                        ja_S_ld_uu[:,l] = ja_S_ld_uu[:,d]
                                        det_ja_S_ld_uu = np.linalg.det(ja_S_ld_uu[0:self.ndocc,0:self.ndocc])

                                        ib_S_kd_uu = mo_overlap_uu.copy()
                                        ib_S_kd_uu[i,:] = ib_S_kd_uu[b,:]
                                        ib_S_kd_uu[:,k] = ib_S_kd_uu[:,d]
                                        det_ib_S_kd_uu = np.linalg.det(ib_S_kd_uu[0:self.ndocc,0:self.ndocc])

                                        ja_S_lc_uu = mo_overlap_uu.copy()
                                        ja_S_lc_uu[j,:] = ja_S_lc_uu[a,:]
                                        ja_S_lc_uu[:,l] = ja_S_lc_uu[:,c]
                                        det_ja_S_lc_uu = np.linalg.det(ja_S_lc_uu[0:self.ndocc,0:self.ndocc])

                                        
                                        # Amplitudes
                                        t_ijab_p = self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc]
                                        t_ijba_p = self.nuc_pos_t2[alpha][i][j][b-self.ndocc][a-self.ndocc]

                                        t_ijab_n = self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc]
                                        t_ijba_n = self.nuc_neg_t2[alpha][i][j][b-self.ndocc][a-self.ndocc]

                                        t_klcd_p = self.mag_pos_t2[beta][k][l][c-self.ndocc][d-self.ndocc]
                                        t_kldc_p = self.mag_pos_t2[beta][k][l][d-self.ndocc][c-self.ndocc]

                                        t_klcd_n = self.mag_neg_t2[beta][k][l][c-self.ndocc][d-self.ndocc]
                                        t_kldc_n = self.mag_neg_t2[beta][k][l][d-self.ndocc][c-self.ndocc]

                                        #I += 0.125 * np.conjugate((t_ijab_p - t_ijab_n) - (t_ijba_p - t_ijba_n)) * (((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed**2)) * det_ijab_S_klcd_uu * det_S_uu - (1/(N_unperturbed**2)) * det_ijab_S_uu * det_S_klcd_uu) + 2 * (t_klcd_p - t_klcd_n) * (1/(N_unperturbed**2)) * det_ijab_S_kc_uu * det_S_ld_uu + 2 * (t_kldc_p - t_kldc_n) * (1/(N_unperturbed**2)) * det_ijab_S_kd_uu * det_S_lc_uu)
                                        #I += 0.125 * np.conjugate((t_ijab_p - t_ijab_n)) * (((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed**2)) * det_ijab_S_klcd_uu * det_S_uu  - (1/(N_unperturbed**2)) * det_ijab_S_uu * det_S_klcd_uu) + 2 * (t_klcd_p - t_klcd_n) * (1/(N_unperturbed**2)) * det_ijab_S_kc_uu * det_S_ld_uu + 2 * (t_kldc_p - t_kldc_n) * (1/(N_unperturbed**2)) * det_ijab_S_kd_uu * det_S_lc_uu)
                                        #I += 0.125 * np.conjugate((t_ijba_p - t_ijba_n)) * (((t_klcd_p - t_klcd_n) - (t_kldc_p - t_kldc_n)) * ((1/(N_unperturbed**2)) * det_ijab_S_klcd_uu * det_S_uu  - (1/(N_unperturbed**2)) * det_ijab_S_uu * det_S_klcd_uu) + 2 * (t_klcd_p - t_klcd_n) * (1/(N_unperturbed**2)) * det_ijab_S_kc_uu * det_S_ld_uu + 2 * (t_kldc_p - t_kldc_n) * (1/(N_unperturbed**2)) * det_ijab_S_kd_uu * det_S_lc_uu)


        return I * (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength))




