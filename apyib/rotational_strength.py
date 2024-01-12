"""Contains the class and functions associated with computing the rotational strength for VCD calculations by finite difference at the Hartree-Fock level of theory."""

import psi4
import numpy as np
import math
import itertools as it
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

    # Computes the permutations required for the Hartree-Fock wavefunction.
    def compute_perms(self):
        det = np.arange(0, self.ndocc)
        size = len(det)
        permutation = []
        parity = []

        heaperm(det, size, permutation, parity)
        return parity, permutation

    # Computes the overlap between two Hartree-Fock wavefunctions.
    def compute_hf_overlap(self, mo_overlap):
        det = np.arange(0, self.ndocc)
        mo_prod = 1
        hf_overlap = 0
        for n in it.permutations(det, r=None):
            perm = list(n)
            par = list(n)
            sign = perm_parity(par)
            for i in range(0, self.ndocc):
                mo_prod *= mo_overlap[det[i], perm[i]]
            hf_overlap += sign * mo_prod
            mo_prod = 1

        return hf_overlap

    # Computes the Hartree-Fock AATs.
    def compute_hf_aat(self, alpha, beta):
        # Compute phase corrected wavefunctions.
        pc_nuc_pos_wfn = compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha])
        pc_nuc_neg_wfn = compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha])
        pc_mag_pos_wfn = compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_pos_basis[beta], self.mag_pos_wfn[beta])
        pc_mag_neg_wfn = compute_phase(self.ndocc, self.nbf, self.unperturbed_basis, self.unperturbed_wfn, self.mag_neg_basis[beta], self.mag_neg_wfn[beta])

        # Compute molecular orbital overlaps with phase correction applied.
        mo_overlap_pp = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], pc_nuc_pos_wfn, self.mag_pos_basis[beta], pc_mag_pos_wfn)
        mo_overlap_np = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], pc_nuc_neg_wfn, self.mag_pos_basis[beta], pc_mag_pos_wfn)
        mo_overlap_pn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_pos_basis[alpha], pc_nuc_pos_wfn, self.mag_neg_basis[beta], pc_mag_neg_wfn)
        mo_overlap_nn = compute_mo_overlap(self.ndocc, self.nbf, self.nuc_neg_basis[alpha], pc_nuc_neg_wfn, self.mag_neg_basis[beta], pc_mag_neg_wfn)

        # Compute Hartree-Fock overlaps.
        hf_pp = np.linalg.det(mo_overlap_pp[0:self.ndocc, 0:self.ndocc])
        hf_np = np.linalg.det(mo_overlap_np[0:self.ndocc, 0:self.ndocc])
        hf_pn = np.linalg.det(mo_overlap_pn[0:self.ndocc, 0:self.ndocc])
        hf_nn = np.linalg.det(mo_overlap_nn[0:self.ndocc, 0:self.ndocc]) 

        # Compute the AAT.
        I = (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_pp - hf_np - hf_pn + hf_nn)

        return 2*I



    # Computes the terms in the CID AATs. Note that the phase corrections were applied in the finite difference code.
    def compute_cid_aat(self, alpha, beta):
        # Compute MO overlaps.
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

        # Compute the HF term of the CID AAT.
        I = (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * (hf_pgpg - hf_pgng - hf_ngpg + hf_ngng)
        I = 0

        # Compute the terms including only one doubly excited determinant in either the bra or ket. 
        for i in range(0, self.ndocc):
            for j in range(i+1, self.ndocc):
                for a in range(self.ndocc, self.nbf):
                    for b in range(a+1, self.nbf):

                        # Swap the rows for orbital substituion in the bra.
                        # < ijab | d0/dH >
                        mo_overlap_egp = mo_overlap_up.copy()
                        mo_overlap_egp[[i, a],:] = mo_overlap_egp[[a, i],:]
                        mo_overlap_egp[[j, b],:] = mo_overlap_egp[[b, j],:]

                        mo_overlap_egn = mo_overlap_un.copy()
                        mo_overlap_egn[[i, a],:] = mo_overlap_egn[[a, i],:]
                        mo_overlap_egn[[j, b],:] = mo_overlap_egn[[b, j],:]

                        # < dijab/dR | d0/dH >
                        mo_overlap_epgp = mo_overlap_pp.copy()
                        mo_overlap_epgp[[i, a],:] = mo_overlap_epgp[[a, i],:]
                        mo_overlap_epgp[[j, b],:] = mo_overlap_epgp[[b, j],:]

                        mo_overlap_epgn = mo_overlap_pn.copy()
                        mo_overlap_epgn[[i, a],:] = mo_overlap_epgn[[a, i],:]
                        mo_overlap_epgn[[j, b],:] = mo_overlap_epgn[[b, j],:]

                        mo_overlap_engp = mo_overlap_np.copy()
                        mo_overlap_engp[[i, a],:] = mo_overlap_engp[[a, i],:]
                        mo_overlap_engp[[j, b],:] = mo_overlap_engp[[b, j],:]

                        mo_overlap_engn = mo_overlap_nn.copy()
                        mo_overlap_engn[[i, a],:] = mo_overlap_engn[[a, i],:]
                        mo_overlap_engn[[j, b],:] = mo_overlap_engn[[b, j],:]

                        # Swap the columns for orbital substituion in the ket.
                        # < d0/dR | ijab >
                        mo_overlap_gpe = mo_overlap_pu.copy()
                        mo_overlap_gpe[:,[i, a]] = mo_overlap_gpe[:,[a, i]]
                        mo_overlap_gpe[:,[j, b]] = mo_overlap_gpe[:,[b, j]]

                        mo_overlap_gne = mo_overlap_nu.copy()
                        mo_overlap_gne[:,[i, a]] = mo_overlap_gne[:,[a, i]]
                        mo_overlap_gne[:,[j, b]] = mo_overlap_gne[:,[b, j]]

                        # < d0/dR | dijab/dH >
                        mo_overlap_gpep = mo_overlap_pp.copy()
                        mo_overlap_gpep[:,[i, a]] = mo_overlap_gpep[:,[a, i]]
                        mo_overlap_gpep[:,[j, b]] = mo_overlap_gpep[:,[b, j]]

                        mo_overlap_gpen = mo_overlap_pn.copy()
                        mo_overlap_gpen[:,[i, a]] = mo_overlap_gpen[:,[a, i]]
                        mo_overlap_gpen[:,[j, b]] = mo_overlap_gpen[:,[b, j]]

                        mo_overlap_gnep = mo_overlap_np.copy()
                        mo_overlap_gnep[:,[i, a]] = mo_overlap_gnep[:,[a, i]]
                        mo_overlap_gnep[:,[j, b]] = mo_overlap_gnep[:,[b, j]]

                        mo_overlap_gnen = mo_overlap_nn.copy()
                        mo_overlap_gnen[:,[i, a]] = mo_overlap_gnen[:,[a, i]]
                        mo_overlap_gnen[:,[j, b]] = mo_overlap_gnen[:,[b, j]]

                        # Compute determinant overlap.
                        det_overlap_egp = np.linalg.det(mo_overlap_egp[0:self.ndocc, 0:self.ndocc])
                        det_overlap_egn = np.linalg.det(mo_overlap_egn[0:self.ndocc, 0:self.ndocc])

                        det_overlap_epgp = np.linalg.det(mo_overlap_epgp[0:self.ndocc, 0:self.ndocc])
                        det_overlap_epgn = np.linalg.det(mo_overlap_epgn[0:self.ndocc, 0:self.ndocc])
                        det_overlap_engp = np.linalg.det(mo_overlap_engp[0:self.ndocc, 0:self.ndocc])
                        det_overlap_engn = np.linalg.det(mo_overlap_engn[0:self.ndocc, 0:self.ndocc])

                        det_overlap_gpe = np.linalg.det(mo_overlap_gpe[0:self.ndocc, 0:self.ndocc])
                        det_overlap_gne = np.linalg.det(mo_overlap_gne[0:self.ndocc, 0:self.ndocc])

                        det_overlap_gpep = np.linalg.det(mo_overlap_gpep[0:self.ndocc, 0:self.ndocc])
                        det_overlap_gpen = np.linalg.det(mo_overlap_gpen[0:self.ndocc, 0:self.ndocc])
                        det_overlap_gnep = np.linalg.det(mo_overlap_gnep[0:self.ndocc, 0:self.ndocc])
                        det_overlap_gnen = np.linalg.det(mo_overlap_gnen[0:self.ndocc, 0:self.ndocc])

                        # Compute contribution of this component to the AAT.
                        I += 0.5 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc] - self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc])) * (det_overlap_egp - det_overlap_egn)

                        I += 0.5 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * np.conjugate(self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]) * (det_overlap_epgp - det_overlap_epgn - det_overlap_engp + det_overlap_engn)

                        I += 0.5 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * (self.mag_pos_t2[beta][i][j][a-self.ndocc][b-self.ndocc] - self.mag_neg_t2[beta][i][j][a-self.ndocc][b-self.ndocc]) * (det_overlap_gpe - det_overlap_gne)

                        I += 0.5 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc] * (det_overlap_gpep - det_overlap_gpen - det_overlap_gnep + det_overlap_gnen)

                        for k in range(0, self.ndocc):
                            for l in range(k+1, self.ndocc):
                                for c in range(self.ndocc, self.nbf):
                                    for d in range(c+1, self.nbf):

                                        # Swap the rows and columns for orbital substituion in the bra and ket.
                                        # < ijab | klcd >
                                        mo_overlap_ee = mo_overlap_uu.copy()
                                        mo_overlap_ee[[i, a],:] = mo_overlap_ee[[a, i],:]
                                        mo_overlap_ee[[j, b],:] = mo_overlap_ee[[b, j],:]
                                        mo_overlap_ee[:,[k, c]] = mo_overlap_ee[:,[c, k]]
                                        mo_overlap_ee[:,[l, d]] = mo_overlap_ee[:,[d, l]]

                                        # < ijab | dklcd/dH >
                                        mo_overlap_eep = mo_overlap_up.copy()
                                        mo_overlap_eep[[i, a],:] = mo_overlap_eep[[a, i],:]
                                        mo_overlap_eep[[j, b],:] = mo_overlap_eep[[b, j],:]
                                        mo_overlap_eep[:,[k, c]] = mo_overlap_eep[:,[c, k]]
                                        mo_overlap_eep[:,[l, d]] = mo_overlap_eep[:,[d, l]]

                                        mo_overlap_een = mo_overlap_un.copy()
                                        mo_overlap_een[[i, a],:] = mo_overlap_een[[a, i],:]
                                        mo_overlap_een[[j, b],:] = mo_overlap_een[[b, j],:]
                                        mo_overlap_een[:,[k, c]] = mo_overlap_een[:,[c, k]]
                                        mo_overlap_een[:,[l, d]] = mo_overlap_een[:,[d, l]]

                                        # < dijab/dR | klcd >
                                        mo_overlap_epe = mo_overlap_pu.copy()
                                        mo_overlap_epe[[i, a],:] = mo_overlap_epe[[a, i],:]
                                        mo_overlap_epe[[j, b],:] = mo_overlap_epe[[b, j],:]
                                        mo_overlap_epe[:,[k, c]] = mo_overlap_epe[:,[c, k]]
                                        mo_overlap_epe[:,[l, d]] = mo_overlap_epe[:,[d, l]]

                                        mo_overlap_ene = mo_overlap_nu.copy()
                                        mo_overlap_ene[[i, a],:] = mo_overlap_ene[[a, i],:]
                                        mo_overlap_ene[[j, b],:] = mo_overlap_ene[[b, j],:]
                                        mo_overlap_ene[:,[k, c]] = mo_overlap_ene[:,[c, k]]
                                        mo_overlap_ene[:,[l, d]] = mo_overlap_ene[:,[d, l]]

                                        # < dijab/dR | dklcd/dH >
                                        mo_overlap_epep = mo_overlap_pp.copy()
                                        mo_overlap_epep[[i, a],:] = mo_overlap_epep[[a, i],:]
                                        mo_overlap_epep[[j, b],:] = mo_overlap_epep[[b, j],:]
                                        mo_overlap_epep[:,[k, c]] = mo_overlap_epep[:,[c, k]]
                                        mo_overlap_epep[:,[l, d]] = mo_overlap_epep[:,[d, l]]

                                        mo_overlap_epen = mo_overlap_pn.copy()
                                        mo_overlap_epen[[i, a],:] = mo_overlap_epen[[a, i],:]
                                        mo_overlap_epen[[j, b],:] = mo_overlap_epen[[b, j],:]
                                        mo_overlap_epen[:,[k, c]] = mo_overlap_epen[:,[c, k]]
                                        mo_overlap_epen[:,[l, d]] = mo_overlap_epen[:,[d, l]]

                                        mo_overlap_enep = mo_overlap_np.copy()
                                        mo_overlap_enep[[i, a],:] = mo_overlap_enep[[a, i],:]
                                        mo_overlap_enep[[j, b],:] = mo_overlap_enep[[b, j],:]
                                        mo_overlap_enep[:,[k, c]] = mo_overlap_enep[:,[c, k]]
                                        mo_overlap_enep[:,[l, d]] = mo_overlap_enep[:,[d, l]]

                                        mo_overlap_enen = mo_overlap_nn.copy()
                                        mo_overlap_enen[[i, a],:] = mo_overlap_enen[[a, i],:]
                                        mo_overlap_enen[[j, b],:] = mo_overlap_enen[[b, j],:]
                                        mo_overlap_enen[:,[k, c]] = mo_overlap_enen[:,[c, k]]
                                        mo_overlap_enen[:,[l, d]] = mo_overlap_enen[:,[d, l]]

                                        # Compute determinant overlap.
                                        det_overlap_ee = np.linalg.det(mo_overlap_ee[0:self.ndocc, 0:self.ndocc])

                                        det_overlap_eep = np.linalg.det(mo_overlap_eep[0:self.ndocc, 0:self.ndocc])
                                        det_overlap_een = np.linalg.det(mo_overlap_een[0:self.ndocc, 0:self.ndocc])

                                        det_overlap_epe = np.linalg.det(mo_overlap_epe[0:self.ndocc, 0:self.ndocc])
                                        det_overlap_ene = np.linalg.det(mo_overlap_ene[0:self.ndocc, 0:self.ndocc])

                                        det_overlap_epep = np.linalg.det(mo_overlap_epep[0:self.ndocc, 0:self.ndocc])
                                        det_overlap_epen = np.linalg.det(mo_overlap_epen[0:self.ndocc, 0:self.ndocc])
                                        det_overlap_enep = np.linalg.det(mo_overlap_enep[0:self.ndocc, 0:self.ndocc])
                                        det_overlap_enen = np.linalg.det(mo_overlap_enen[0:self.ndocc, 0:self.ndocc])

                                        # Compute contribution of this component to the AAT.
                                        I += 0.25 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc] - self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc])) * (self.mag_pos_t2[beta][k][l][c-self.ndocc][d-self.ndocc] - self.mag_neg_t2[beta][k][l][c-self.ndocc][d-self.ndocc]) * det_overlap_ee

                                        I += 0.25 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * np.conjugate((self.nuc_pos_t2[alpha][i][j][a-self.ndocc][b-self.ndocc] - self.nuc_neg_t2[alpha][i][j][a-self.ndocc][b-self.ndocc])) * self.unperturbed_t2[k][l][c-self.ndocc][d-self.ndocc] * (det_overlap_eep - det_overlap_een)

                                        I += 0.25 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * np.conjugate(self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]) * (self.mag_pos_t2[beta][k][l][c-self.ndocc][d-self.ndocc] - self.mag_neg_t2[beta][k][l][c-self.ndocc][d-self.ndocc]) * (det_overlap_epe - det_overlap_ene)

                                        I += 0.25 * 1/(4 * self.nuc_pert_strength * self.mag_pert_strength) * np.conjugate(self.unperturbed_t2[i][j][a-self.ndocc][b-self.ndocc]) * self.unperturbed_t2[k][l][c-self.ndocc][d-self.ndocc] * (det_overlap_epep - det_overlap_epen - det_overlap_enep + det_overlap_enen)




        return 2*I

