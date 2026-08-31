"""Finite-difference velocity-gauge atomic polar tensors (VG APT) for RHF, MP2, CID, and CISD."""

import psi4
import numpy as np
import math
import itertools as it
import time
import gc
import opt_einsum as oe
from apyib.utils import compute_mo_overlap
from apyib.utils import compute_so_overlap
from apyib.hamiltonian import Hamiltonian
from apyib.energy import energy
from apyib.hf_wfn import hf_wfn
from apyib.fin_diff import finite_difference



class VG_APT(object):
    """Velocity-gauge atomic polar tensor object computed by finite difference."""

    def __init__(self, parameters, wfn, unperturbed_wfn, unperturbed_basis, unperturbed_T,
                 nuc_pos_wfn, nuc_neg_wfn, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
                 mom_pos_wfn, mom_neg_wfn, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
                 nuc_pert_strength, mom_pert_strength):
        """Store displaced-geometry wavefunctions for finite-difference VG APT evaluation.

        Parameters
        ----------
        parameters : dict
            Calculation parameters.
        wfn : hf_wfn
            Reference (unperturbed) RHF wavefunction.
        unperturbed_wfn : ndarray
            MO coefficients at the reference geometry.
        unperturbed_basis : psi4.core.BasisSet
            Basis set at the reference geometry.
        unperturbed_T : list
            Amplitude list at the reference geometry.
        nuc_pos_wfn, nuc_neg_wfn : list of ndarray
            MO coefficients at +/- nuclear displacements.
        nuc_pos_basis, nuc_neg_basis : list of psi4.core.BasisSet
            Basis sets at +/- nuclear displacements.
        nuc_pos_T, nuc_neg_T : list
            Amplitude lists at +/- nuclear displacements.
        mom_pos_wfn, mom_neg_wfn : list of ndarray
            MO coefficients at +/- vector potential (momentum) perturbations.
        mom_pos_basis, mom_neg_basis : list of psi4.core.BasisSet
            Basis sets at +/- momentum perturbations (same as reference for field perturbations).
        mom_pos_T, mom_neg_T : list
            Amplitude lists at +/- momentum perturbations.
        nuc_pert_strength : float
            Step size for nuclear coordinate displacements.
        mom_pert_strength : float
            Step size for vector potential field perturbations.
        """
        self.nuc_pos_wfn = nuc_pos_wfn
        self.nuc_neg_wfn = nuc_neg_wfn
        self.nuc_pos_T = nuc_pos_T
        self.nuc_neg_T = nuc_neg_T

        self.mom_pos_wfn = mom_pos_wfn
        self.mom_neg_wfn = mom_neg_wfn
        self.mom_pos_T = mom_pos_T
        self.mom_neg_T = mom_neg_T

        self.nuc_pert_strength = nuc_pert_strength
        self.mom_pert_strength = mom_pert_strength

        self.unperturbed_wfn = unperturbed_wfn
        self.unperturbed_T = unperturbed_T

        H = Hamiltonian(parameters)
        natom = H.molecule.natom()
        self.nbf = wfn.nbf
        self.ndocc = wfn.ndocc
        self.nfzc = wfn.H.basis_set.n_frozen_core()
        self.parameters = parameters

        # Store nuclear charges for the nuclear contribution to VG APT.
        _, _, _, Z, _ = H.molecule.to_arrays()
        self.Z = Z

        # Compute MO overlaps.
        if parameters['method'] != 'RHF':
            # < psi | psi >
            mo_overlap_uu = compute_mo_overlap(self.ndocc, self.nbf, unperturbed_basis, self.unperturbed_wfn, unperturbed_basis, self.unperturbed_wfn)
            if parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
                so_overlap_uu = compute_so_overlap(self.nbf, mo_overlap_uu)
                self.overlap_uu = so_overlap_uu
            else:
                self.overlap_uu = mo_overlap_uu

            # < psi | dpsi/dA >
            self.overlap_up = []
            self.overlap_un = []
            for beta in range(3):
                mo_overlap_up = compute_mo_overlap(self.ndocc, self.nbf, unperturbed_basis, self.unperturbed_wfn, mom_pos_basis[beta], self.mom_pos_wfn[beta])
                mo_overlap_un = compute_mo_overlap(self.ndocc, self.nbf, unperturbed_basis, self.unperturbed_wfn, mom_neg_basis[beta], self.mom_neg_wfn[beta])
                if parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
                    so_overlap_up = compute_so_overlap(self.nbf, mo_overlap_up)
                    so_overlap_un = compute_so_overlap(self.nbf, mo_overlap_un)
                    self.overlap_up.append(so_overlap_up)
                    self.overlap_un.append(so_overlap_un)
                else:
                    self.overlap_up.append(mo_overlap_up)
                    self.overlap_un.append(mo_overlap_un)

            # < dpsi/dR | psi >
            self.overlap_pu = []
            self.overlap_nu = []
            for alpha in range(3*natom):
                mo_overlap_pu = compute_mo_overlap(self.ndocc, self.nbf, nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], unperturbed_basis, self.unperturbed_wfn)
                mo_overlap_nu = compute_mo_overlap(self.ndocc, self.nbf, nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], unperturbed_basis, self.unperturbed_wfn)
                if parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
                    so_overlap_pu = compute_so_overlap(self.nbf, mo_overlap_pu)
                    so_overlap_nu = compute_so_overlap(self.nbf, mo_overlap_nu)
                    self.overlap_pu.append(so_overlap_pu)
                    self.overlap_nu.append(so_overlap_nu)
                else:
                    self.overlap_pu.append(mo_overlap_pu)
                    self.overlap_nu.append(mo_overlap_nu)

        # < dpsi/dR | dpsi/dA >
        alpha = 3 * natom
        beta = 3
        self.overlap_pp = [[[] for _ in range(beta)] for _ in range(alpha)]
        self.overlap_pn = [[[] for _ in range(beta)] for _ in range(alpha)]
        self.overlap_np = [[[] for _ in range(beta)] for _ in range(alpha)]
        self.overlap_nn = [[[] for _ in range(beta)] for _ in range(alpha)]
        for alpha in range(3*natom):
            for beta in range(3):
                mo_overlap_pp = compute_mo_overlap(self.ndocc, self.nbf, nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], mom_pos_basis[beta], self.mom_pos_wfn[beta])
                mo_overlap_pn = compute_mo_overlap(self.ndocc, self.nbf, nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], mom_neg_basis[beta], self.mom_neg_wfn[beta])
                mo_overlap_np = compute_mo_overlap(self.ndocc, self.nbf, nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], mom_pos_basis[beta], self.mom_pos_wfn[beta])
                mo_overlap_nn = compute_mo_overlap(self.ndocc, self.nbf, nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], mom_neg_basis[beta], self.mom_neg_wfn[beta])
                if parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
                    self.overlap_pp[alpha][beta] = compute_so_overlap(self.nbf, mo_overlap_pp)
                    self.overlap_pn[alpha][beta] = compute_so_overlap(self.nbf, mo_overlap_pn)
                    self.overlap_np[alpha][beta] = compute_so_overlap(self.nbf, mo_overlap_np)
                    self.overlap_nn[alpha][beta] = compute_so_overlap(self.nbf, mo_overlap_nn)
                else:
                    self.overlap_pp[alpha][beta] = mo_overlap_pp
                    self.overlap_pn[alpha][beta] = mo_overlap_pn
                    self.overlap_np[alpha][beta] = mo_overlap_np
                    self.overlap_nn[alpha][beta] = mo_overlap_nn



    def compute_SO_det(self, overlap, bra_indices, ket_indices):
        """Compute the occupied-block determinant after swapping bra/ket rows and columns."""
        nocc = 2 * self.ndocc
        S = overlap.copy()
        for x in range(0, len(bra_indices), 2):
            S[[bra_indices[x], bra_indices[x+1]],:] = S[[bra_indices[x+1], bra_indices[x]],:]
        for y in range(0, len(ket_indices), 2):
            S[:,[ket_indices[y], ket_indices[y+1]]] = S[:,[ket_indices[y+1], ket_indices[y]]]
        det_S = np.linalg.det(S[0:nocc,0:nocc])
        return det_S



    def compute_normalization(self, alpha, beta, normalization):
        """Compute wavefunction normalization factors for the given displacement pair."""
        if self.parameters['method'] == 'RHF' or normalization == 'intermediate':
            N = 1
            N_np = 1
            N_nn = 1
            N_mp = 1
            N_mn = 1

        elif normalization == 'full' and self.parameters['method'] == 'CISD_SO':
            N = 1 / np.sqrt(self.unperturbed_T[0]**2 + oe.contract('ia,ia->', np.conjugate(self.unperturbed_T[1]), self.unperturbed_T[1]) + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0]**2 + oe.contract('ia,ia->', np.conjugate(self.nuc_pos_T[alpha][1]), self.nuc_pos_T[alpha][1]) + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0]**2 + oe.contract('ia,ia->', np.conjugate(self.nuc_neg_T[alpha][1]), self.nuc_neg_T[alpha][1]) + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]))
            N_mp = 1 / np.sqrt(self.mom_pos_T[beta][0]**2 + oe.contract('ia,ia->', np.conjugate(self.mom_pos_T[beta][1]), self.mom_pos_T[beta][1]) + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.mom_pos_T[beta][2]), self.mom_pos_T[beta][2]))
            N_mn = 1 / np.sqrt(self.mom_neg_T[beta][0]**2 + oe.contract('ia,ia->', np.conjugate(self.mom_neg_T[beta][1]), self.mom_neg_T[beta][1]) + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.mom_neg_T[beta][2]), self.mom_neg_T[beta][2]))

        elif normalization == 'full' and self.parameters['method'] != 'CISD_SO':
            N = 1 / np.sqrt(self.unperturbed_T[0]**2 + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0]**2 + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0]**2 + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]))
            N_mp = 1 / np.sqrt(self.mom_pos_T[beta][0]**2 + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.mom_pos_T[beta][2]), self.mom_pos_T[beta][2]))
            N_mn = 1 / np.sqrt(self.mom_neg_T[beta][0]**2 + 0.25 * oe.contract('ijab,ijab->', np.conjugate(self.mom_neg_T[beta][2]), self.mom_neg_T[beta][2]))

        return N, N_np, N_nn, N_mp, N_mn



    def compute_SO_I_00(self, alpha, beta, normalization):
        """Return the HF (reference/reference) VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        if self.parameters['method'] == 'RHF':
            self.overlap_pp[alpha][beta] = compute_so_overlap(self.nbf, self.overlap_pp[alpha][beta])
            self.overlap_pn[alpha][beta] = compute_so_overlap(self.nbf, self.overlap_pn[alpha][beta])
            self.overlap_np[alpha][beta] = compute_so_overlap(self.nbf, self.overlap_np[alpha][beta])
            self.overlap_nn[alpha][beta] = compute_so_overlap(self.nbf, self.overlap_nn[alpha][beta])

        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [], [])
        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [], [])
        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [], [])
        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [], [])

        I = det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_0D(self, alpha, beta, normalization):
        """Return the reference/doubles VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for j in range(0, nocc):
                    for b in range(nocc, nbf):
                        det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [], [i,a,j,b])
                        det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [], [i,a,j,b])
                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [], [i,a,j,b])
                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [], [i,a,j,b])
                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [], [i,a,j,b])
                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [], [i,a,j,b])

                        t2_dA = self.mom_pos_T[beta][2][i][j][a-nocc][b-nocc] - self.mom_neg_T[beta][2][i][j][a-nocc][b-nocc]
                        t2 = self.unperturbed_T[2][i][j][a-nocc][b-nocc]

                        I += 0.25 * t2_dA * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                        I += 0.25 * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_D0(self, alpha, beta, normalization):
        """Return the doubles/reference VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for j in range(0, nocc):
                    for b in range(nocc, nbf):
                        det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a,j,b], [])
                        det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a,j,b], [])
                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a,j,b], [])
                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a,j,b], [])
                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a,j,b], [])
                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a,j,b], [])

                        t2_dR = np.conjugate(self.nuc_pos_T[alpha][2][i][j][a-nocc][b-nocc] - self.nuc_neg_T[alpha][2][i][j][a-nocc][b-nocc])
                        t2_conj = np.conjugate(self.unperturbed_T[2][i][j][a-nocc][b-nocc])

                        I += 0.25 * t2_dR * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                        I += 0.25 * t2_conj * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_DD(self, alpha, beta, normalization):
        """Return the doubles/doubles VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for j in range(0, nocc):
                    for b in range(nocc, nbf):
                        for k in range(0, nocc):
                            for c in range(nocc, nbf):
                                for l in range(0, nocc):
                                    for d in range(nocc, nbf):
                                        det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a,j,b], [k,c,l,d])
                                        det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a,j,b], [k,c,l,d])
                                        det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a,j,b], [k,c,l,d])
                                        det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a,j,b], [k,c,l,d])
                                        det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a,j,b], [k,c,l,d])
                                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a,j,b], [k,c,l,d])
                                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a,j,b], [k,c,l,d])
                                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a,j,b], [k,c,l,d])
                                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a,j,b], [k,c,l,d])

                                        t2_dR = np.conjugate(self.nuc_pos_T[alpha][2][i][j][a-nocc][b-nocc] - self.nuc_neg_T[alpha][2][i][j][a-nocc][b-nocc])
                                        t2_dA = self.mom_pos_T[beta][2][k][l][c-nocc][d-nocc] - self.mom_neg_T[beta][2][k][l][c-nocc][d-nocc]
                                        t2_conj = np.conjugate(self.unperturbed_T[2][i][j][a-nocc][b-nocc])
                                        t2 = self.unperturbed_T[2][k][l][c-nocc][d-nocc]

                                        I += 0.25**2 * t2_dR * t2_dA * (det_S_uu * N * N)
                                        I += 0.25**2 * t2_dR * t2 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                        I += 0.25**2 * t2_conj * t2_dA * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                        I += 0.25**2 * t2_conj * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_0S(self, alpha, beta, normalization):
        """Return the reference/singles VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [], [i,a])
                det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [], [i,a])
                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [], [i,a])
                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [], [i,a])
                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [], [i,a])
                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [], [i,a])

                t1_dA = self.mom_pos_T[beta][1][i][a-nocc] - self.mom_neg_T[beta][1][i][a-nocc]
                t1 = self.unperturbed_T[1][i][a-nocc]

                I += t1_dA * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                I += t1 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_S0(self, alpha, beta, normalization):
        """Return the singles/reference VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a], [])
                det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a], [])
                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a], [])
                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a], [])
                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a], [])
                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a], [])

                t1_dR = np.conjugate(self.nuc_pos_T[alpha][1][i][a-nocc] - self.nuc_neg_T[alpha][1][i][a-nocc])
                t1_conj = np.conjugate(self.unperturbed_T[1][i][a-nocc])

                I += t1_dR * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                I += t1_conj * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_SS(self, alpha, beta, normalization):
        """Return the singles/singles VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for k in range(0, nocc):
                    for c in range(nocc, nbf):
                        det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a], [k,c])
                        det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a], [k,c])
                        det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a], [k,c])
                        det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a], [k,c])
                        det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a], [k,c])
                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a], [k,c])
                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a], [k,c])
                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a], [k,c])
                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a], [k,c])

                        t1_dR = np.conjugate(self.nuc_pos_T[alpha][1][i][a-nocc] - self.nuc_neg_T[alpha][1][i][a-nocc])
                        t1_dA = self.mom_pos_T[beta][1][k][c-nocc] - self.mom_neg_T[beta][1][k][c-nocc]
                        t1_conj = np.conjugate(self.unperturbed_T[1][i][a-nocc])
                        t1 = self.unperturbed_T[1][k][c-nocc]

                        I += t1_dR * t1_dA * (det_S_uu * N * N)
                        I += t1_dR * t1 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                        I += t1_conj * t1_dA * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                        I += t1_conj * t1 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_SD(self, alpha, beta, normalization):
        """Return the singles/doubles VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for k in range(0, nocc):
                    for c in range(nocc, nbf):
                        for l in range(0, nocc):
                            for d in range(nocc, nbf):
                                det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a], [k,c,l,d])
                                det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a], [k,c,l,d])
                                det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a], [k,c,l,d])
                                det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a], [k,c,l,d])
                                det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a], [k,c,l,d])
                                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a], [k,c,l,d])
                                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a], [k,c,l,d])
                                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a], [k,c,l,d])
                                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a], [k,c,l,d])

                                t1_dR = np.conjugate(self.nuc_pos_T[alpha][1][i][a-nocc] - self.nuc_neg_T[alpha][1][i][a-nocc])
                                t2_dA = self.mom_pos_T[beta][2][k][l][c-nocc][d-nocc] - self.mom_neg_T[beta][2][k][l][c-nocc][d-nocc]
                                t1_conj = np.conjugate(self.unperturbed_T[1][i][a-nocc])
                                t2 = self.unperturbed_T[2][k][l][c-nocc][d-nocc]

                                I += 0.25 * t1_dR * t2_dA * (det_S_uu * N * N)
                                I += 0.25 * t1_dR * t2 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                I += 0.25 * t1_conj * t2_dA * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                I += 0.25 * t1_conj * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_I_DS(self, alpha, beta, normalization):
        """Return the doubles/singles VG APT contribution in the SO basis."""
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for j in range(0, nocc):
                    for b in range(nocc, nbf):
                        for k in range(0, nocc):
                            for c in range(nocc, nbf):
                                det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a,j,b], [k,c])
                                det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a,j,b], [k,c])
                                det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a,j,b], [k,c])
                                det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a,j,b], [k,c])
                                det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a,j,b], [k,c])
                                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a,j,b], [k,c])
                                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a,j,b], [k,c])
                                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a,j,b], [k,c])
                                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a,j,b], [k,c])

                                t2_dR = np.conjugate(self.nuc_pos_T[alpha][2][i][j][a-nocc][b-nocc] - self.nuc_neg_T[alpha][2][i][j][a-nocc][b-nocc])
                                t1_dA = self.mom_pos_T[beta][1][k][c-nocc] - self.mom_neg_T[beta][1][k][c-nocc]
                                t2_conj = np.conjugate(self.unperturbed_T[2][i][j][a-nocc][b-nocc])
                                t1 = self.unperturbed_T[1][k][c-nocc]

                                I += 0.25 * t2_dR * t1_dA * (det_S_uu * N * N)
                                I += 0.25 * t2_dR * t1 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                I += 0.25 * t2_conj * t1_dA * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                I += 0.25 * t2_conj * t1 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag



    def compute_SO_vg_apts(self, alpha, beta, normalization='full'):
        """Compute one VG APT element (alpha, beta) in the spin-orbital basis."""
        t0 = time.time()
        I_00 = self.compute_SO_I_00(alpha, beta, normalization)

        if self.parameters['method'] != 'RHF':
            I_0D = self.compute_SO_I_0D(alpha, beta, normalization)
            I_D0 = self.compute_SO_I_D0(alpha, beta, normalization)
            I_DD = self.compute_SO_I_DD(alpha, beta, normalization)
        else:
            I_0D = 0
            I_D0 = 0
            I_DD = 0

        if self.parameters['method'] == 'CISD_SO':
            I_0S = self.compute_SO_I_0S(alpha, beta, normalization)
            I_S0 = self.compute_SO_I_S0(alpha, beta, normalization)
            I_SS = self.compute_SO_I_SS(alpha, beta, normalization)
            I_SD = self.compute_SO_I_SD(alpha, beta, normalization)
            I_DS = self.compute_SO_I_DS(alpha, beta, normalization)
        else:
            I_0S = 0
            I_S0 = 0
            I_SS = 0
            I_SD = 0
            I_DS = 0

        I = I_00 + I_0D + I_D0 + I_DD + I_0S + I_S0 + I_SS + I_SD + I_DS

        # Nuclear contribution: Z_lambda * delta(alpha_direction, beta_field).
        lambda_ = alpha // 3
        direction = alpha % 3
        nuclear_term = self.Z[lambda_] if direction == beta else 0.0

        t1 = time.time()
        print(f"VG APT element computed in {t1-t0} seconds.")

        return I + nuclear_term



    def compute_all_dets(self, overlap):
        """Pre-compute all row/column-swapped occupied-block determinants needed for spatial VG APTs."""
        nf = self.nfzc
        no = self.ndocc
        nv = self.nbf - self.ndocc

        S = np.copy(overlap)

        det_ia_S = np.zeros((no-nf, nv), dtype='cdouble')
        det_S_kc = np.zeros((no-nf, nv), dtype='cdouble')
        det_iajb_S = np.zeros((no-nf, nv, no-nf, nv), dtype='cdouble')
        det_S_kcld = np.zeros((no-nf, nv, no-nf, nv), dtype='cdouble')
        det_ia_S_kc = np.zeros((no-nf, nv, no-nf, nv), dtype='cdouble')
        det_iajb_S_kc = np.zeros((no-nf, nv, no-nf, nv, no-nf, nv), dtype='cdouble')
        det_ia_S_kcld = np.zeros((no-nf, nv, no-nf, nv, no-nf, nv), dtype='cdouble')
        det_iajb_S_kcld = np.zeros((no-nf, nv, no-nf, nv, no-nf, nv, no-nf, nv), dtype='cdouble')

        det_S = np.linalg.det(S[0:no, 0:no])

        for i in range(nf, no):
            for a in range(0, nv):
                ia_S = np.copy(S)
                ia_S[[i, a + no],:] = ia_S[[a + no, i],:]
                det_ia_S[i-nf][a] = np.linalg.det(ia_S[0:no, 0:no])
                S_kc = np.copy(S)
                S_kc[:,[i, a + no]] = S_kc[:,[a + no, i]]
                det_S_kc[i-nf][a] = np.linalg.det(S_kc[0:no, 0:no])

                for j in range(i+1, no):
                    for b in range(a+1, nv):
                        iajb_S = np.copy(ia_S)
                        iajb_S[[j, b + no],:] = iajb_S[[b + no, j],:]
                        det_iajb_S[i-nf][a][j-nf][b] = np.linalg.det(iajb_S[0:no, 0:no])
                        S_kcld = np.copy(S_kc)
                        S_kcld[:,[j, b + no]] = S_kcld[:,[b + no, j]]
                        det_S_kcld[i-nf][a][j-nf][b] = np.linalg.det(S_kcld[0:no, 0:no])

                        for k in range(nf, no):
                            for c in range(0, nv):
                                iajb_S_kc = np.copy(iajb_S)
                                iajb_S_kc[:,[k, c + no]] = iajb_S_kc[:,[c + no, k]]
                                det_iajb_S_kc[i-nf][a][j-nf][b][k-nf][c] = np.linalg.det(iajb_S_kc[0:no, 0:no])
                                ia_S_kcld = np.copy(S_kcld)
                                ia_S_kcld[[k, c + no],:] = ia_S_kcld[[c + no, k],:]
                                det_ia_S_kcld[i-nf][a][j-nf][b][k-nf][c] = np.linalg.det(ia_S_kcld[0:no, 0:no])

                                for l in range(k+1, no):
                                    for d in range(c+1, nv):
                                        iajb_S_kcld = np.copy(iajb_S_kc)
                                        iajb_S_kcld[:,[l, d + no]] = iajb_S_kcld[:,[d + no, l]]
                                        det_iajb_S_kcld[i-nf][a][j-nf][b][k-nf][c][l-nf][d] = np.linalg.det(iajb_S_kcld[0:no, 0:no])

                for k in range(nf, no):
                    for c in range(0, nv):
                        ia_S_kc = np.copy(ia_S)
                        ia_S_kc[:,[k, c + no]] = ia_S_kc[:,[c + no, k]]
                        det_ia_S_kc[i-nf][a][k-nf][c] = np.linalg.det(ia_S_kc[0:no, 0:no])

        det_iajb_S = det_iajb_S - np.swapaxes(det_iajb_S,0,2) - np.swapaxes(det_iajb_S,1,3) + np.swapaxes(np.swapaxes(det_iajb_S,0,2),1,3)
        det_S_kcld = det_S_kcld - np.swapaxes(det_S_kcld,0,2) - np.swapaxes(det_S_kcld,1,3) + np.swapaxes(np.swapaxes(det_S_kcld,0,2),1,3)
        det_iajb_S_kc = det_iajb_S_kc - np.swapaxes(det_iajb_S_kc,0,2) - np.swapaxes(det_iajb_S_kc,1,3) + np.swapaxes(np.swapaxes(det_iajb_S_kc,0,2),1,3)
        det_ia_S_kcld = det_ia_S_kcld - np.swapaxes(det_ia_S_kcld,0,2) - np.swapaxes(det_ia_S_kcld,1,3) + np.swapaxes(np.swapaxes(det_ia_S_kcld,0,2),1,3)
        det_ia_S_kcld = np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(det_ia_S_kcld,0,4),1,5),2,4),3,5)
        det_iajb_S_kcld = det_iajb_S_kcld - np.swapaxes(det_iajb_S_kcld,0,2) - np.swapaxes(det_iajb_S_kcld,1,3) - np.swapaxes(det_iajb_S_kcld,4,6) - np.swapaxes(det_iajb_S_kcld,5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,4,6),5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),4,6) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,1,3),5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,1,3),4,6) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3),4,6) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3),5,7) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),4,6),5,7) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,1,3),4,6),5,7) + np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3),4,6),5,7)

        return det_S, det_ia_S, det_S_kc, det_iajb_S, det_S_kcld, det_ia_S_kc, det_iajb_S_kc, det_ia_S_kcld, det_iajb_S_kcld



    def compute_spatial_vg_apts(self, alpha, beta, normalization='full'):
        """Compute one VG APT element (alpha, beta) in the spatial orbital basis."""
        t0 = time.time()
        if self.parameters['method'] == 'RHF' or normalization == 'intermediate':
            N = 1
            N_np = 1
            N_nn = 1
            N_mp = 1
            N_mn = 1
        elif normalization == 'full' and self.parameters['method'] != 'CISD':
            N = 1 / np.sqrt(self.unperturbed_T[0] + (2*oe.contract('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]) - oe.contract('ijab,ijba->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2])))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0] + (2*oe.contract('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]) - oe.contract('ijab,ijba->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2])))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0] + (2*oe.contract('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]) - oe.contract('ijab,ijba->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2])))
            N_mp = 1 / np.sqrt(self.mom_pos_T[beta][0] + (2*oe.contract('ijab,ijab->', np.conjugate(self.mom_pos_T[beta][2]), self.mom_pos_T[beta][2]) - oe.contract('ijab,ijba->', np.conjugate(self.mom_pos_T[beta][2]), self.mom_pos_T[beta][2])))
            N_mn = 1 / np.sqrt(self.mom_neg_T[beta][0] + (2*oe.contract('ijab,ijab->', np.conjugate(self.mom_neg_T[beta][2]), self.mom_neg_T[beta][2]) - oe.contract('ijab,ijba->', np.conjugate(self.mom_neg_T[beta][2]), self.mom_neg_T[beta][2])))
        elif normalization == 'full' and self.parameters['method'] == 'CISD':
            N = 1 / np.sqrt(self.unperturbed_T[0] + 2*oe.contract('ia,ia->', np.conjugate(self.unperturbed_T[1]), self.unperturbed_T[1]) + (2*oe.contract('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]) - oe.contract('ijab,ijba->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2])))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0] + 2*oe.contract('ia,ia->', np.conjugate(self.nuc_pos_T[alpha][1]), self.nuc_pos_T[alpha][1]) + (2*oe.contract('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]) - oe.contract('ijab,ijba->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2])))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0] + 2*oe.contract('ia,ia->', np.conjugate(self.nuc_neg_T[alpha][1]), self.nuc_neg_T[alpha][1]) + (2*oe.contract('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]) - oe.contract('ijab,ijba->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2])))
            N_mp = 1 / np.sqrt(self.mom_pos_T[beta][0] + 2*oe.contract('ia,ia->', np.conjugate(self.mom_pos_T[beta][1]), self.mom_pos_T[beta][1]) + (2*oe.contract('ijab,ijab->', np.conjugate(self.mom_pos_T[beta][2]), self.mom_pos_T[beta][2]) - oe.contract('ijab,ijba->', np.conjugate(self.mom_pos_T[beta][2]), self.mom_pos_T[beta][2])))
            N_mn = 1 / np.sqrt(self.mom_neg_T[beta][0] + 2*oe.contract('ia,ia->', np.conjugate(self.mom_neg_T[beta][1]), self.mom_neg_T[beta][1]) + (2*oe.contract('ijab,ijab->', np.conjugate(self.mom_neg_T[beta][2]), self.mom_neg_T[beta][2]) - oe.contract('ijab,ijba->', np.conjugate(self.mom_neg_T[beta][2]), self.mom_neg_T[beta][2])))

        # Compute the HF VG APT contribution (squared determinant of occupied-block overlap).
        S_pp = np.linalg.det(self.overlap_pp[alpha][beta][0:self.ndocc,0:self.ndocc])**2
        S_pn = np.linalg.det(self.overlap_pn[alpha][beta][0:self.ndocc,0:self.ndocc])**2
        S_np = np.linalg.det(self.overlap_np[alpha][beta][0:self.ndocc,0:self.ndocc])**2
        S_nn = np.linalg.det(self.overlap_nn[alpha][beta][0:self.ndocc,0:self.ndocc])**2

        I_00 = S_pp * N_np * N_mp - S_pn * N_np * N_mn - S_np * N_nn * N_mp + S_nn * N_nn * N_mn
        I_S0 = 0
        I_0S = 0
        I_D0 = 0
        I_0D = 0
        I_SS = 0
        I_DS = 0
        I_SD = 0
        I_DD = 0

        if self.parameters['method'] != 'RHF':
            if self.parameters['method'] == 'CISD':
                t1 = N * self.unperturbed_T[1]
                t1_dA = N_mp * self.mom_pos_T[beta][1] - N_mn * self.mom_neg_T[beta][1]
                t1_conj = np.conjugate(t1)
                t1_dR = np.conjugate(N_np * self.nuc_pos_T[alpha][1] - N_nn * self.nuc_neg_T[alpha][1])

            t2 = N * self.unperturbed_T[2]
            t2_dA = N_mp * self.mom_pos_T[beta][2] - N_mn * self.mom_neg_T[beta][2]
            t2_conj = np.conjugate(t2)
            t2_dR = np.conjugate(N_np * self.nuc_pos_T[alpha][2] - N_nn * self.nuc_neg_T[alpha][2])

            # Unperturbed / Unperturbed
            S_uu, ia_S_uu, S_kc_uu, iajb_S_uu, S_kcld_uu, ia_S_kc_uu, iajb_S_kc_uu, ia_S_kcld_uu, iajb_S_kcld_uu = self.compute_all_dets(self.overlap_uu)

            if self.parameters['method'] == 'CISD':
                I_SS += 2 * oe.contract('ia,kc,iakc->', t1_dR, t1_dA, ia_S_kc_uu) * S_uu
                I_SS += 2 * oe.contract('ia,ia->', t1_dR, ia_S_uu) * oe.contract('kc,kc->', t1_dA, S_kc_uu)
                I_DS += 0.5 * oe.contract('ijab,kc,iajbkc->', t2_dR - np.swapaxes(t2_dR, 2, 3), t1_dA, iajb_S_kc_uu) * S_uu
                I_DS += 0.5 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_uu) * oe.contract('kc,kc->', t1_dA, S_kc_uu)
                I_DS += 2 * oe.contract('ijab,kc,iakc,jb->', t2_dR, t1_dA, ia_S_kc_uu, ia_S_uu)
                I_SD += 0.5 * oe.contract('ia,klcd,iakcld->', t1_dR, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_kcld_uu) * S_uu
                I_SD += 0.5 * oe.contract('ia,ia->', t1_dR, ia_S_uu) * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_uu)
                I_SD += 2 * oe.contract('ia,klcd,iakc,ld->', t1_dR, t2_dA, ia_S_kc_uu, S_kc_uu)

            I_DD += 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2_dA - np.swapaxes(t2_dA, 2, 3), iajb_S_kcld_uu) * S_uu
            I_DD += 0.125 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_uu) * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_uu)
            I_DD += 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2_dA, iajb_S_kc_uu, S_kc_uu)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_dR, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_kcld_uu, ia_S_uu)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_dR, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_uu, ia_S_kcld_uu)
            I_DD += 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_dR, t2_dA, ia_S_kc_uu, ia_S_kc_uu)
            del S_uu; del ia_S_uu; del S_kc_uu; del iajb_S_uu; del S_kcld_uu; del ia_S_kc_uu; del iajb_S_kc_uu; del ia_S_kcld_uu; del iajb_S_kcld_uu
            gc.collect()

            # Unperturbed / Positive
            S_up, ia_S_up, S_kc_up, iajb_S_up, S_kcld_up, ia_S_kc_up, iajb_S_kc_up, ia_S_kcld_up, iajb_S_kcld_up = self.compute_all_dets(self.overlap_up[beta])

            if self.parameters['method'] == 'CISD':
                I_S0 += 2 * oe.contract('ia,ia->', t1_dR, ia_S_up) * S_up * N_mp
                I_D0 += 0.5 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * S_up
                I_D0 += oe.contract('jb,jb->', oe.contract('ijab,ia->jb', t2_dR, ia_S_up), ia_S_up)
                I_SS += 2 * oe.contract('ia,kc,iakc->', t1_dR, t1, ia_S_kc_up) * S_up
                I_SS += 2 * oe.contract('ia,ia->', t1_dR, ia_S_up) * oe.contract('kc,kc->', t1, S_kc_up)
                I_DS += 0.5 * oe.contract('ijab,kc,iajbkc->', t2_dR - np.swapaxes(t2_dR, 2, 3), t1, iajb_S_kc_up) * S_up
                I_DS += 0.5 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * oe.contract('kc,kc->', t1, S_kc_up)
                I_DS += 2 * oe.contract('ijab,kc,iakc,jb->', t2_dR, t1, ia_S_kc_up, ia_S_up)
                I_SD += 0.5 * oe.contract('ia,klcd,iakcld->', t1_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_up) * S_up
                I_SD += 0.5 * oe.contract('ia,ia->', t1_dR, ia_S_up) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_up)
                I_SD += 2 * oe.contract('ia,klcd,iakc,ld->', t1_dR, t2, ia_S_kc_up, S_kc_up)

            I_DD += 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_up) * S_up
            I_DD += 0.125 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_up)
            I_DD += 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2, iajb_S_kc_up, S_kc_up)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_up, ia_S_up)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_up, ia_S_kcld_up)
            I_DD += 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_dR, t2, ia_S_kc_up, ia_S_kc_up)
            del S_up; del ia_S_up; del S_kc_up; del iajb_S_up; del S_kcld_up; del ia_S_kc_up; del iajb_S_kc_up; del ia_S_kcld_up; del iajb_S_kcld_up
            gc.collect()

            # Unperturbed / Negative
            S_un, ia_S_un, S_kc_un, iajb_S_un, S_kcld_un, ia_S_kc_un, iajb_S_kc_un, ia_S_kcld_un, iajb_S_kcld_un = self.compute_all_dets(self.overlap_un[beta])

            if self.parameters['method'] == 'CISD':
                I_S0 -= 2 * oe.contract('ia,ia->', t1_dR, ia_S_un) * S_un * N_mn
                I_D0 -= 0.5 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * S_un
                I_D0 -= oe.contract('jb,jb->', oe.contract('ijab,ia->jb', t2_dR, ia_S_un), ia_S_un)
                I_SS -= 2 * oe.contract('ia,kc,iakc->', t1_dR, t1, ia_S_kc_un) * S_un
                I_SS -= 2 * oe.contract('ia,ia->', t1_dR, ia_S_un) * oe.contract('kc,kc->', t1, S_kc_un)
                I_DS -= 0.5 * oe.contract('ijab,kc,iajbkc->', t2_dR - np.swapaxes(t2_dR, 2, 3), t1, iajb_S_kc_un) * S_un
                I_DS -= 0.5 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * oe.contract('kc,kc->', t1, S_kc_un)
                I_DS -= 2 * oe.contract('ijab,kc,iakc,jb->', t2_dR, t1, ia_S_kc_un, ia_S_un)
                I_SD -= 0.5 * oe.contract('ia,klcd,iakcld->', t1_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_un) * S_un
                I_SD -= 0.5 * oe.contract('ia,ia->', t1_dR, ia_S_un) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_un)
                I_SD -= 2 * oe.contract('ia,klcd,iakc,ld->', t1_dR, t2, ia_S_kc_un, S_kc_un)

            I_DD -= 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_un) * S_un
            I_DD -= 0.125 * oe.contract('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_un)
            I_DD -= 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2, iajb_S_kc_un, S_kc_un)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_un, ia_S_un)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_un, ia_S_kcld_un)
            I_DD -= 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_dR, t2, ia_S_kc_un, ia_S_kc_un)
            del S_un; del ia_S_un; del S_kc_un; del iajb_S_un; del S_kcld_un; del ia_S_kc_un; del iajb_S_kc_un; del ia_S_kcld_un; del iajb_S_kcld_un
            gc.collect()

            # Positive / Unperturbed
            S_pu, ia_S_pu, S_kc_pu, iajb_S_pu, S_kcld_pu, ia_S_kc_pu, iajb_S_kc_pu, ia_S_kcld_pu, iajb_S_kcld_pu = self.compute_all_dets(self.overlap_pu[alpha])

            if self.parameters['method'] == 'CISD':
                I_0S += 2 * oe.contract('kc,kc->', t1_dA, S_kc_pu) * S_pu * N_np
                I_0D += 0.5 * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_pu) * S_pu
                I_0D += oe.contract('ld,ld->', oe.contract('klcd,kc->ld', t2_dA, S_kc_pu), S_kc_pu)
                I_SS += 2 * oe.contract('ia,kc,iakc->', t1_conj, t1_dA, ia_S_kc_pu) * S_pu
                I_SS += 2 * oe.contract('ia,ia->', t1_conj, ia_S_pu) * oe.contract('kc,kc->', t1_dA, S_kc_pu)
                I_SD += 0.5 * oe.contract('ia,klcd,iakcld->', t1_conj, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_kcld_pu) * S_pu
                I_SD += 0.5 * oe.contract('ia,ia->', t1_conj, ia_S_pu) * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_pu)
                I_SD += 2 * oe.contract('ia,klcd,iakc,ld->', t1_conj, t2_dA, ia_S_kc_pu, S_kc_pu)
                I_DS += 0.5 * oe.contract('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1_dA, iajb_S_kc_pu) * S_pu
                I_DS += 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pu) * oe.contract('kc,kc->', t1_dA, S_kc_pu)
                I_DS += 2 * oe.contract('ijab,kc,iakc,jb->', t2_conj, t1_dA, ia_S_kc_pu, ia_S_pu)

            I_DD += 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dA - np.swapaxes(t2_dA, 2, 3), iajb_S_kcld_pu) * S_pu
            I_DD += 0.125 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pu) * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_pu)
            I_DD += 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dA, iajb_S_kc_pu, S_kc_pu)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_conj, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_kcld_pu, ia_S_pu)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_conj, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_pu, ia_S_kcld_pu)
            I_DD += 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_conj, t2_dA, ia_S_kc_pu, ia_S_kc_pu)
            del S_pu; del ia_S_pu; del S_kc_pu; del iajb_S_pu; del S_kcld_pu; del ia_S_kc_pu; del iajb_S_kc_pu; del ia_S_kcld_pu; del iajb_S_kcld_pu
            gc.collect()

            # Negative / Unperturbed
            S_nu, ia_S_nu, S_kc_nu, iajb_S_nu, S_kcld_nu, ia_S_kc_nu, iajb_S_kc_nu, ia_S_kcld_nu, iajb_S_kcld_nu = self.compute_all_dets(self.overlap_nu[alpha])

            if self.parameters['method'] == 'CISD':
                I_0S -= 2 * oe.contract('kc,kc->', t1_dA, S_kc_nu) * S_nu * N_nn
                I_0D -= 0.5 * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_nu) * S_nu
                I_0D -= oe.contract('ld,ld->', oe.contract('klcd,kc->ld', t2_dA, S_kc_nu), S_kc_nu)
                I_SS -= 2 * oe.contract('ia,kc,iakc->', t1_conj, t1_dA, ia_S_kc_nu) * S_nu
                I_SS -= 2 * oe.contract('ia,ia->', t1_conj, ia_S_nu) * oe.contract('kc,kc->', t1_dA, S_kc_nu)
                I_SD -= 0.5 * oe.contract('ia,klcd,iakcld->', t1_conj, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_kcld_nu) * S_nu
                I_SD -= 0.5 * oe.contract('ia,ia->', t1_conj, ia_S_nu) * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_nu)
                I_SD -= 2 * oe.contract('ia,klcd,iakc,ld->', t1_conj, t2_dA, ia_S_kc_nu, S_kc_nu)
                I_DS -= 0.5 * oe.contract('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1_dA, iajb_S_kc_nu) * S_nu
                I_DS -= 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nu) * oe.contract('kc,kc->', t1_dA, S_kc_nu)
                I_DS -= 2 * oe.contract('ijab,kc,iakc,jb->', t2_conj, t1_dA, ia_S_kc_nu, ia_S_nu)

            I_DD -= 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dA - np.swapaxes(t2_dA, 2, 3), iajb_S_kcld_nu) * S_nu
            I_DD -= 0.125 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nu) * oe.contract('klcd,kcld->', t2_dA - np.swapaxes(t2_dA, 2, 3), S_kcld_nu)
            I_DD -= 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dA, iajb_S_kc_nu, S_kc_nu)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_conj, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_kcld_nu, ia_S_nu)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_conj, t2_dA - np.swapaxes(t2_dA, 2, 3), ia_S_nu, ia_S_kcld_nu)
            I_DD -= 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_conj, t2_dA, ia_S_kc_nu, ia_S_kc_nu)
            del S_nu; del ia_S_nu; del S_kc_nu; del iajb_S_nu; del S_kcld_nu; del ia_S_kc_nu; del iajb_S_kc_nu; del ia_S_kcld_nu; del iajb_S_kcld_nu
            gc.collect()

            # Positive / Positive
            S_pp, ia_S_pp, S_kc_pp, iajb_S_pp, S_kcld_pp, ia_S_kc_pp, iajb_S_kc_pp, ia_S_kcld_pp, iajb_S_kcld_pp = self.compute_all_dets(self.overlap_pp[alpha][beta])

            if self.parameters['method'] == 'CISD':
                I_0S += 2 * oe.contract('kc,kc->', t1, S_kc_pp) * S_pp * N_np
                I_S0 += 2 * oe.contract('ia,ia->', t1_conj, ia_S_pp) * S_pp * N_mp
                I_0D += 0.5 * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp) * S_pp
                I_0D += oe.contract('ld,ld->', oe.contract('klcd,kc->ld', t2, S_kc_pp), S_kc_pp)
                I_D0 += 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * S_pp
                I_D0 += oe.contract('jb,jb->', oe.contract('ijab,ia->jb', t2_conj, ia_S_pp), ia_S_pp)
                I_SS += 2 * oe.contract('ia,kc,iakc->', t1_conj, t1, ia_S_kc_pp) * S_pp
                I_SS += 2 * oe.contract('ia,ia->', t1_conj, ia_S_pp) * oe.contract('kc,kc->', t1, S_kc_pp)
                I_SD += 0.5 * oe.contract('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pp) * S_pp
                I_SD += 0.5 * oe.contract('ia,ia->', t1_conj, ia_S_pp) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp)
                I_SD += 2 * oe.contract('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_pp, S_kc_pp)
                I_DS += 0.5 * oe.contract('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_pp) * S_pp
                I_DS += 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * oe.contract('kc,kc->', t1, S_kc_pp)
                I_DS += 2 * oe.contract('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_pp, ia_S_pp)

            I_DD += 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_pp) * S_pp
            I_DD += 0.125 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp)
            I_DD += 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_pp, S_kc_pp)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pp, ia_S_pp)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_pp, ia_S_kcld_pp)
            I_DD += 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_pp, ia_S_kc_pp)
            del S_pp; del ia_S_pp; del S_kc_pp; del iajb_S_pp; del S_kcld_pp; del ia_S_kc_pp; del iajb_S_kc_pp; del ia_S_kcld_pp; del iajb_S_kcld_pp
            gc.collect()

            # Positive / Negative
            S_pn, ia_S_pn, S_kc_pn, iajb_S_pn, S_kcld_pn, ia_S_kc_pn, iajb_S_kc_pn, ia_S_kcld_pn, iajb_S_kcld_pn = self.compute_all_dets(self.overlap_pn[alpha][beta])

            if self.parameters['method'] == 'CISD':
                I_0S -= 2 * oe.contract('kc,kc->', t1, S_kc_pn) * S_pn * N_np
                I_S0 -= 2 * oe.contract('ia,ia->', t1_conj, ia_S_pn) * S_pn * N_mn
                I_0D -= 0.5 * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn) * S_pn
                I_0D -= oe.contract('ld,ld->', oe.contract('klcd,kc->ld', t2, S_kc_pn), S_kc_pn)
                I_D0 -= 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * S_pn
                I_D0 -= oe.contract('jb,jb->', oe.contract('ijab,ia->jb', t2_conj, ia_S_pn), ia_S_pn)
                I_SS -= 2 * oe.contract('ia,kc,iakc->', t1_conj, t1, ia_S_kc_pn) * S_pn
                I_SS -= 2 * oe.contract('ia,ia->', t1_conj, ia_S_pn) * oe.contract('kc,kc->', t1, S_kc_pn)
                I_SD -= 0.5 * oe.contract('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pn) * S_pn
                I_SD -= 0.5 * oe.contract('ia,ia->', t1_conj, ia_S_pn) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn)
                I_SD -= 2 * oe.contract('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_pn, S_kc_pn)
                I_DS -= 0.5 * oe.contract('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_pn) * S_pn
                I_DS -= 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * oe.contract('kc,kc->', t1, S_kc_pn)
                I_DS -= 2 * oe.contract('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_pn, ia_S_pn)

            I_DD -= 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_pn) * S_pn
            I_DD -= 0.125 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn)
            I_DD -= 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_pn, S_kc_pn)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pn, ia_S_pn)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_pn, ia_S_kcld_pn)
            I_DD -= 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_pn, ia_S_kc_pn)
            del S_pn; del ia_S_pn; del S_kc_pn; del iajb_S_pn; del S_kcld_pn; del ia_S_kc_pn; del iajb_S_kc_pn; del ia_S_kcld_pn; del iajb_S_kcld_pn
            gc.collect()

            # Negative / Positive
            S_np, ia_S_np, S_kc_np, iajb_S_np, S_kcld_np, ia_S_kc_np, iajb_S_kc_np, ia_S_kcld_np, iajb_S_kcld_np = self.compute_all_dets(self.overlap_np[alpha][beta])

            if self.parameters['method'] == 'CISD':
                I_0S -= 2 * oe.contract('kc,kc->', t1, S_kc_np) * S_np * N_nn
                I_S0 -= 2 * oe.contract('ia,ia->', t1_conj, ia_S_np) * S_np * N_mp
                I_0D -= 0.5 * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np) * S_np
                I_0D -= oe.contract('ld,ld->', oe.contract('klcd,kc->ld', t2, S_kc_np), S_kc_np)
                I_D0 -= 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * S_np
                I_D0 -= oe.contract('jb,jb->', oe.contract('ijab,ia->jb', t2_conj, ia_S_np), ia_S_np)
                I_SS -= 2 * oe.contract('ia,kc,iakc->', t1_conj, t1, ia_S_kc_np) * S_np
                I_SS -= 2 * oe.contract('ia,ia->', t1_conj, ia_S_np) * oe.contract('kc,kc->', t1, S_kc_np)
                I_SD -= 0.5 * oe.contract('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_np) * S_np
                I_SD -= 0.5 * oe.contract('ia,ia->', t1_conj, ia_S_np) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np)
                I_SD -= 2 * oe.contract('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_np, S_kc_np)
                I_DS -= 0.5 * oe.contract('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_np) * S_np
                I_DS -= 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * oe.contract('kc,kc->', t1, S_kc_np)
                I_DS -= 2 * oe.contract('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_np, ia_S_np)

            I_DD -= 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_np) * S_np
            I_DD -= 0.125 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np)
            I_DD -= 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_np, S_kc_np)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_np, ia_S_np)
            I_DD -= 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_np, ia_S_kcld_np)
            I_DD -= 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_np, ia_S_kc_np)
            del S_np; del ia_S_np; del S_kc_np; del iajb_S_np; del S_kcld_np; del ia_S_kc_np; del iajb_S_kc_np; del ia_S_kcld_np; del iajb_S_kcld_np
            gc.collect()

            # Negative / Negative
            S_nn, ia_S_nn, S_kc_nn, iajb_S_nn, S_kcld_nn, ia_S_kc_nn, iajb_S_kc_nn, ia_S_kcld_nn, iajb_S_kcld_nn = self.compute_all_dets(self.overlap_nn[alpha][beta])

            if self.parameters['method'] == 'CISD':
                I_0S += 2 * oe.contract('kc,kc->', t1, S_kc_nn) * S_nn * N_nn
                I_S0 += 2 * oe.contract('ia,ia->', t1_conj, ia_S_nn) * S_nn * N_mn
                I_0D += 0.5 * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn) * S_nn
                I_0D += oe.contract('ld,ld->', oe.contract('klcd,kc->ld', t2, S_kc_nn), S_kc_nn)
                I_D0 += 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * S_nn
                I_D0 += oe.contract('jb,jb->', oe.contract('ijab,ia->jb', t2_conj, ia_S_nn), ia_S_nn)
                I_SS += 2 * oe.contract('ia,kc,iakc->', t1_conj, t1, ia_S_kc_nn) * S_nn
                I_SS += 2 * oe.contract('ia,ia->', t1_conj, ia_S_nn) * oe.contract('kc,kc->', t1, S_kc_nn)
                I_SD += 0.5 * oe.contract('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_nn) * S_nn
                I_SD += 0.5 * oe.contract('ia,ia->', t1_conj, ia_S_nn) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn)
                I_SD += 2 * oe.contract('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_nn, S_kc_nn)
                I_DS += 0.5 * oe.contract('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_nn) * S_nn
                I_DS += 0.5 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * oe.contract('kc,kc->', t1, S_kc_nn)
                I_DS += 2 * oe.contract('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_nn, ia_S_nn)

            I_DD += 0.125 * oe.contract('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_nn) * S_nn
            I_DD += 0.125 * oe.contract('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * oe.contract('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn)
            I_DD += 0.125 * 4 * oe.contract('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_nn, S_kc_nn)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_nn, ia_S_nn)
            I_DD += 0.125 * 2 * oe.contract('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_nn, ia_S_kcld_nn)
            I_DD += 0.125 * 2 * 4 * oe.contract('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_nn, ia_S_kc_nn)
            del S_nn; del ia_S_nn; del S_kc_nn; del iajb_S_nn; del S_kcld_nn; del ia_S_kc_nn; del iajb_S_kc_nn; del ia_S_kcld_nn; del iajb_S_kcld_nn
            gc.collect()

        I = I_00 + I_0D + I_D0 + I_DD + I_0S + I_S0 + I_SS + I_SD + I_DS

        # Nuclear contribution: Z_lambda * delta(displacement_direction, field_direction).
        lambda_ = alpha // 3
        direction = alpha % 3
        nuclear_term = self.Z[lambda_] if direction == beta else 0.0

        t1_time = time.time()
        print(f"VG APT element computed in {t1_time-t0} seconds.")

        return (1 / (2 * self.nuc_pert_strength * self.mom_pert_strength)) * I.imag + nuclear_term
