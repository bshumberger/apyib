"""Contains the class and functions associated with computing the rotational strength for VCD calculations by finite difference at the Hartree-Fock, MP2, CID, and CISD levels of theory."""

import psi4
import numpy as np
import math
import itertools as it
import time
import gc
from apyib.utils import compute_mo_overlap
from apyib.utils import compute_so_overlap
from apyib.hamiltonian import Hamiltonian
from apyib.energy import energy
from apyib.hf_wfn import hf_wfn
from apyib.fin_diff import finite_difference



class AAT(object):
    """
    The atomic axial tensor object computed by finite difference.
    """
    def __init__(self, parameters, wfn, unperturbed_wfn, unperturbed_basis, unperturbed_T, nuc_pos_wfn, nuc_neg_wfn, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_wfn, mag_neg_wfn, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T, nuc_pert_strength, mag_pert_strength):

        # Basis sets and wavefunctions from calculations with respect to nuclear displacements.
        self.nuc_pos_wfn = nuc_pos_wfn
        self.nuc_neg_wfn = nuc_neg_wfn
        self.nuc_pos_T = nuc_pos_T
        self.nuc_neg_T = nuc_neg_T

        # Basis sets and wavefunctions from calculations with respect to magnetic field perturbations.
        self.mag_pos_wfn = mag_pos_wfn
        self.mag_neg_wfn = mag_neg_wfn
        self.mag_pos_T = mag_pos_T
        self.mag_neg_T = mag_neg_T

        # Components required for finite difference AATs.
        self.nuc_pert_strength = nuc_pert_strength
        self.mag_pert_strength = mag_pert_strength

        # Components required for unperturbed wavefunction.
        self.unperturbed_wfn = unperturbed_wfn
        self.unperturbed_T = unperturbed_T

        # Components required for permutations.
        H = Hamiltonian(parameters)
        natom = H.molecule.natom()
        self.nbf = wfn.nbf
        self.ndocc = wfn.ndocc
        self.nfzc = wfn.H.basis_set.n_frozen_core()
        self.parameters = parameters

        # Compute MO overlaps.
        if parameters['method'] != 'RHF':
            # < psi | psi >
            mo_overlap_uu = compute_mo_overlap(self.ndocc, self.nbf, unperturbed_basis, self.unperturbed_wfn, unperturbed_basis, self.unperturbed_wfn)
            if parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
                so_overlap_uu = compute_so_overlap(self.nbf, mo_overlap_uu)
                self.overlap_uu = so_overlap_uu
            else:
                self.overlap_uu = mo_overlap_uu

            # < psi | dpsi/dH >
            self.overlap_up = []
            self.overlap_un = []
            for beta in range(3):
                mo_overlap_up = compute_mo_overlap(self.ndocc, self.nbf, unperturbed_basis, self.unperturbed_wfn, mag_pos_basis[beta], self.mag_pos_wfn[beta])
                mo_overlap_un = compute_mo_overlap(self.ndocc, self.nbf, unperturbed_basis, self.unperturbed_wfn, mag_neg_basis[beta], self.mag_neg_wfn[beta])
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

        # < dpsi/dR | dpsi/dH >
        alpha = 3 * natom
        beta = 3
        self.overlap_pp = [[[] for _ in range(beta)] for _ in range(alpha)]
        self.overlap_pn = [[[] for _ in range(beta)] for _ in range(alpha)] 
        self.overlap_np = [[[] for _ in range(beta)] for _ in range(alpha)] 
        self.overlap_nn = [[[] for _ in range(beta)] for _ in range(alpha)] 
        for alpha in range(3*natom):
            for beta in range(3):
                mo_overlap_pp = compute_mo_overlap(self.ndocc, self.nbf, nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], mag_pos_basis[beta], self.mag_pos_wfn[beta])
                mo_overlap_pn = compute_mo_overlap(self.ndocc, self.nbf, nuc_pos_basis[alpha], self.nuc_pos_wfn[alpha], mag_neg_basis[beta], self.mag_neg_wfn[beta])
                mo_overlap_np = compute_mo_overlap(self.ndocc, self.nbf, nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], mag_pos_basis[beta], self.mag_pos_wfn[beta])
                mo_overlap_nn = compute_mo_overlap(self.ndocc, self.nbf, nuc_neg_basis[alpha], self.nuc_neg_wfn[alpha], mag_neg_basis[beta], self.mag_neg_wfn[beta])
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



    # Computes the determinant of the occupied space for some general row and column swap.
    def compute_SO_det(self, overlap, bra_indices, ket_indices):
        nocc = 2 * self.ndocc
        S = overlap.copy()
        for x in range(0, len(bra_indices), 2):
            S[[bra_indices[x], bra_indices[x+1]],:] = S[[bra_indices[x+1], bra_indices[x]],:]
        for y in range(0, len(ket_indices), 2):
            S[:,[ket_indices[y], ket_indices[y+1]]] = S[:,[ket_indices[y+1], ket_indices[y]]]

        det_S = np.linalg.det(S[0:nocc,0:nocc])

        return det_S



    def compute_normalization(self, alpha, beta, normalization):
        # Compute normalization factors.
        if self.parameters['method'] == 'RHF' or normalization == 'intermediate':
            N = 1
            N_np = 1
            N_nn = 1
            N_mp = 1
            N_mn = 1

        elif normalization == 'full' and self.parameters['method'] == 'CISD_SO':
            N = 1 / np.sqrt(self.unperturbed_T[0]**2 + np.einsum('ia,ia->', np.conjugate(self.unperturbed_T[1]), self.unperturbed_T[1]) + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0]**2 + np.einsum('ia,ia->', np.conjugate(self.nuc_pos_T[alpha][1]), self.nuc_pos_T[alpha][1]) + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0]**2 + np.einsum('ia,ia->', np.conjugate(self.nuc_neg_T[alpha][1]), self.nuc_neg_T[alpha][1]) + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]))
            N_mp = 1 / np.sqrt(self.mag_pos_T[beta][0]**2 + np.einsum('ia,ia->', np.conjugate(self.mag_pos_T[beta][1]), self.mag_pos_T[beta][1]) + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2]))
            N_mn = 1 / np.sqrt(self.mag_neg_T[beta][0]**2 + np.einsum('ia,ia->', np.conjugate(self.mag_neg_T[beta][1]), self.mag_neg_T[beta][1]) + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2]))

        elif normalization == 'full' and self.parameters['method'] != 'CISD_SO':
            N = 1 / np.sqrt(self.unperturbed_T[0]**2 + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0]**2 + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0]**2 + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]))
            N_mp = 1 / np.sqrt(self.mag_pos_T[beta][0]**2 + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2]))
            N_mn = 1 / np.sqrt(self.mag_neg_T[beta][0]**2 + 0.25 * np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2]))

        return N, N_np, N_nn, N_mp, N_mn



    def compute_SO_I_00(self, alpha, beta, normalization):
        # Compute the Hartree-Fock determinant in the spin-orbital basis.
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

        # Compute the HF AATs.
        I = det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_0D(self, alpha, beta, normalization):
        # Compute the reference/doubles determinants in the spin-orbital basis.
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for j in range(0, nocc):
                    for b in range(nocc, nbf):
                        # < d0/dR | ijab >
                        det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [], [i,a,j,b])
                        det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [], [i,a,j,b])

                        # < d0/dR | dijab/dH >
                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [], [i,a,j,b])
                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [], [i,a,j,b])
                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [], [i,a,j,b])
                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [], [i,a,j,b])

                        # dt_ijab / dH
                        t2_dH = self.mag_pos_T[beta][2][i][j][a-nocc][b-nocc] - self.mag_neg_T[beta][2][i][j][a-nocc][b-nocc]

                        # t_ijab
                        t2 = self.unperturbed_T[2][i][j][a-nocc][b-nocc]

                        I += 0.25 * t2_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                        I += 0.25 * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_D0(self, alpha, beta, normalization):
        # Compute the doubles/reference determinants in the spin-orbital basis.
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for j in range(0, nocc):
                    for b in range(nocc, nbf):
                        # < ijab | d0/dH >
                        det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a,j,b], [])
                        det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a,j,b], [])

                        # < dijab/dR | d0/dH >
                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a,j,b], [])
                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a,j,b], [])
                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a,j,b], [])
                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a,j,b], [])

                        # dt_ijab / dR
                        t2_dR = np.conjugate(self.nuc_pos_T[alpha][2][i][j][a-nocc][b-nocc] - self.nuc_neg_T[alpha][2][i][j][a-nocc][b-nocc])

                        # t_ijab
                        t2_conj = np.conjugate(self.unperturbed_T[2][i][j][a-nocc][b-nocc])

                        I += 0.25 * t2_dR * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                        I += 0.25 * t2_conj * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag    



    def compute_SO_I_DD(self, alpha, beta, normalization):
        # Compute the doubles/doubles determinants in the spin-orbital basis.
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
                                        # < ijab | klcd >
                                        det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a,j,b], [k,c,l,d])

                                        # < dijab/dR | klcd >
                                        det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a,j,b], [k,c,l,d])
                                        det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a,j,b], [k,c,l,d])

                                        # < ijab | dklcd/dH >
                                        det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a,j,b], [k,c,l,d])
                                        det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a,j,b], [k,c,l,d])

                                        # < dijab/dR | dklcd/dH >
                                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a,j,b], [k,c,l,d])
                                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a,j,b], [k,c,l,d])
                                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a,j,b], [k,c,l,d])
                                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a,j,b], [k,c,l,d])

                                        # dt_ijab / dR
                                        t2_dR = np.conjugate(self.nuc_pos_T[alpha][2][i][j][a-nocc][b-nocc] - self.nuc_neg_T[alpha][2][i][j][a-nocc][b-nocc])

                                        # dt_klcd / dH
                                        t2_dH = self.mag_pos_T[beta][2][k][l][c-nocc][d-nocc] - self.mag_neg_T[beta][2][k][l][c-nocc][d-nocc]

                                        # t_ijab
                                        t2_conj = np.conjugate(self.unperturbed_T[2][i][j][a-nocc][b-nocc])

                                        # t_klcd
                                        t2 = self.unperturbed_T[2][k][l][c-nocc][d-nocc]

                                        I += 0.25**2 * t2_dR * t2_dH * (det_S_uu * N * N)
                                        I += 0.25**2 * t2_dR * t2 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                        I += 0.25**2 * t2_conj * t2_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                        I += 0.25**2 * t2_conj * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_0S(self, alpha, beta, normalization):
        # Compute the reference/singles determinants in the spin-orbital basis.
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0 
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                # < d0/dR | ia >
                det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [], [i,a])
                det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [], [i,a])

                # < d0/dR | dia/dH >
                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [], [i,a])
                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [], [i,a])
                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [], [i,a])
                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [], [i,a])

                # dt_ia / dH
                t1_dH = self.mag_pos_T[beta][1][i][a-nocc] - self.mag_neg_T[beta][1][i][a-nocc]

                # t_ia
                t1 = self.unperturbed_T[1][i][a-nocc]

                I += t1_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                I += t1 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_S0(self, alpha, beta, normalization):
        # Compute the singles/reference determinants in the spin-orbital basis.
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0 
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                # < ia | d0/dH >
                det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a], []) 
                det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a], []) 

                # < dia/dR | d0/dH >
                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a], [])
                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a], [])
                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a], [])
                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a], [])

                # dt_ia / dR
                t1_dR = np.conjugate(self.nuc_pos_T[alpha][1][i][a-nocc] - self.nuc_neg_T[alpha][1][i][a-nocc])

                # t_ia
                t1_conj = np.conjugate(self.unperturbed_T[1][i][a-nocc])

                I += t1_dR * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                I += t1_conj * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_SS(self, alpha, beta, normalization):
        # Compute the singles/singles determinants in the spin-orbital basis.
        N, N_np, N_nn, N_mp, N_mn = self.compute_normalization(alpha, beta, normalization)

        nocc = 2 * self.ndocc
        nbf = 2 * self.nbf
        I = 0
        for i in range(0, nocc):
            for a in range(nocc, nbf):
                for k in range(0, nocc):
                    for c in range(nocc, nbf):
                        # < ia | kc >
                        det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a], [k,c])

                        # < dia/dR | kc >
                        det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a], [k,c])
                        det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a], [k,c])

                        # < ia | dkc/dH >
                        det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a], [k,c])
                        det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a], [k,c])

                        # < dia/dR | dkc/dH >
                        det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a], [k,c])
                        det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a], [k,c])
                        det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a], [k,c])
                        det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a], [k,c])

                        # dt_ia / dR
                        t1_dR = np.conjugate(self.nuc_pos_T[alpha][1][i][a-nocc] - self.nuc_neg_T[alpha][1][i][a-nocc])

                        # dt_kc / dH
                        t1_dH = self.mag_pos_T[beta][1][k][c-nocc] - self.mag_neg_T[beta][1][k][c-nocc]

                        # t_ia
                        t1_conj = np.conjugate(self.unperturbed_T[1][i][a-nocc])

                        # t_kc
                        t1 = self.unperturbed_T[1][k][c-nocc]

                        I += t1_dR * t1_dH * (det_S_uu * N * N)
                        I += t1_dR * t1 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                        I += t1_conj * t1_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                        I += t1_conj * t1 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_SD(self, alpha, beta, normalization):
        # Compute the singles/doubles determinants in the spin-orbital basis.
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
                                # < ia | klcd >
                                det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a], [k,c,l,d])

                                # < dia/dR | klcd >
                                det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a], [k,c,l,d])
                                det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a], [k,c,l,d])

                                # < ia | dklcd/dH >
                                det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a], [k,c,l,d])
                                det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a], [k,c,l,d])

                                # < dia/dR | dklcd/dH >
                                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a], [k,c,l,d])
                                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a], [k,c,l,d])
                                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a], [k,c,l,d])
                                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a], [k,c,l,d])

                                # dt_ia / dR
                                t1_dR = np.conjugate(self.nuc_pos_T[alpha][1][i][a-nocc] - self.nuc_neg_T[alpha][1][i][a-nocc])

                                # dt_klcd / dH
                                t2_dH = self.mag_pos_T[beta][2][k][l][c-nocc][d-nocc] - self.mag_neg_T[beta][2][k][l][c-nocc][d-nocc]

                                # t_ia
                                t1_conj = np.conjugate(self.unperturbed_T[1][i][a-nocc])

                                # t_klcd
                                t2 = self.unperturbed_T[2][k][l][c-nocc][d-nocc]

                                I += 0.25 * t1_dR * t2_dH * (det_S_uu * N * N)
                                I += 0.25 * t1_dR * t2 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                I += 0.25 * t1_conj * t2_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                I += 0.25 * t1_conj * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)


        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_I_DS(self, alpha, beta, normalization):
        # Compute the doubles/singles determinants in the spin-orbital basis.
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
                                # < ijab | kc >
                                det_S_uu = self.compute_SO_det(self.overlap_uu, [i,a,j,b], [k,c])

                                # < dijab/dR | kc >
                                det_S_pu = self.compute_SO_det(self.overlap_pu[alpha], [i,a,j,b], [k,c])
                                det_S_nu = self.compute_SO_det(self.overlap_nu[alpha], [i,a,j,b], [k,c])

                                # < ijab | dkc/dH >
                                det_S_up = self.compute_SO_det(self.overlap_up[beta], [i,a,j,b], [k,c])
                                det_S_un = self.compute_SO_det(self.overlap_un[beta], [i,a,j,b], [k,c])

                                # < dijab/dR | dkc/dH >
                                det_S_pp = self.compute_SO_det(self.overlap_pp[alpha][beta], [i,a,j,b], [k,c])
                                det_S_pn = self.compute_SO_det(self.overlap_pn[alpha][beta], [i,a,j,b], [k,c])
                                det_S_np = self.compute_SO_det(self.overlap_np[alpha][beta], [i,a,j,b], [k,c])
                                det_S_nn = self.compute_SO_det(self.overlap_nn[alpha][beta], [i,a,j,b], [k,c])

                                # dt_ijab / dR
                                t2_dR = np.conjugate(self.nuc_pos_T[alpha][2][i][j][a-nocc][b-nocc] - self.nuc_neg_T[alpha][2][i][j][a-nocc][b-nocc])

                                # dt_kc / dH
                                t1_dH = self.mag_pos_T[beta][1][k][c-nocc] - self.mag_neg_T[beta][1][k][c-nocc]

                                # t_ijab
                                t2_conj = np.conjugate(self.unperturbed_T[2][i][j][a-nocc][b-nocc])

                                # t_kc
                                t1 = self.unperturbed_T[1][k][c-nocc]

                                I += 0.25 * t2_dR * t1_dH * (det_S_uu * N * N)
                                I += 0.25 * t2_dR * t1 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                I += 0.25 * t2_conj * t1_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                I += 0.25 * t2_conj * t1 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_aats(self, alpha, beta, normalization='full'):
        t0 = time.time()
        # Compute the HF term of the AATs.
        I_00 = self.compute_SO_I_00(alpha, beta, normalization)

        # Add doubles contribution (for MP2, CID, and CISD).
        if self.parameters['method'] != 'RHF': 
            I_0D = self.compute_SO_I_0D(alpha, beta, normalization)
            I_D0 = self.compute_SO_I_D0(alpha, beta, normalization)
            I_DD = self.compute_SO_I_DD(alpha, beta, normalization)
        else:
            I_0D = 0
            I_D0 = 0
            I_DD = 0

        # Add singles contribution (for CISD).
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

        t1 =time.time()
        print(f"AAT element computed in {t1-t0} seconds.")

        return I



    def compute_all_dets(self, overlap):
        # Setting up occupied and virtual spaces.
        nf = self.nfzc
        no = self.ndocc
        nv = self.nbf - self.ndocc

        # Computes all the determinants required from the row and column swappings.
        S = np.copy(overlap)

        # Initialize determinant matrices.
        det_ia_S = np.zeros((no-nf, nv), dtype='cdouble')
        det_S_kc = np.zeros((no-nf, nv), dtype='cdouble')
        det_iajb_S = np.zeros((no-nf, nv, no-nf, nv), dtype='cdouble')
        det_S_kcld = np.zeros((no-nf, nv, no-nf, nv), dtype='cdouble')
        det_ia_S_kc = np.zeros((no-nf, nv, no-nf, nv), dtype='cdouble')
        det_iajb_S_kc = np.zeros((no-nf, nv, no-nf, nv, no-nf, nv), dtype='cdouble')
        det_ia_S_kcld = np.zeros((no-nf, nv, no-nf, nv, no-nf, nv), dtype='cdouble')
        det_iajb_S_kcld = np.zeros((no-nf, nv, no-nf, nv, no-nf, nv, no-nf, nv), dtype='cdouble')

        # Determinant of the initial overlap.
        det_S = np.linalg.det(S[0:no, 0:no])

        # Swap indices and compute determinants.
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
        # det_ia_S_kcld = [k][c][l][d][i][a] original
        # det_ia_S_kcld = [i][c][l][d][k][a] np.swapaxes(0,4)
        # det_ia_S_kcld = [i][a][l][d][k][c] np.swapaxes(1,5)
        # det_ia_S_kcld = [i][a][k][d][l][c] np.swapaxes(2,4)
        # det_ia_S_kcld = [i][a][k][c][l][d] np.swapaxes(3,5)
        det_ia_S_kcld = np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(det_ia_S_kcld,0,4),1,5),2,4),3,5)
        det_iajb_S_kcld = det_iajb_S_kcld - np.swapaxes(det_iajb_S_kcld,0,2) - np.swapaxes(det_iajb_S_kcld,1,3) - np.swapaxes(det_iajb_S_kcld,4,6) - np.swapaxes(det_iajb_S_kcld,5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,4,6),5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),4,6) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,1,3),5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),5,7) + np.swapaxes(np.swapaxes(det_iajb_S_kcld,1,3),4,6) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3),4,6) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3),5,7) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),4,6),5,7) - np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,1,3),4,6),5,7) + np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(det_iajb_S_kcld,0,2),1,3),4,6),5,7)

        #print("det_: ", det_S)
        #print("det_ia: ", det_ia_S)
        #print("det_kc: ", det_S_kc)
        #print("det_iajb: ", det_iajb_S)
        #print("det_kcld: ", det_S_kcld)
        #print("det_iakc: ", det_ia_S_kc)
        #print("det_iajbkc: ", det_iajb_S_kc)
        #print("det_iakcld: ", det_ia_S_kcld)
        #print("det_iajbkcld: ", det_iajb_S_kcld)

        return det_S, det_ia_S, det_S_kc, det_iajb_S, det_S_kcld, det_ia_S_kc, det_iajb_S_kc, det_ia_S_kcld, det_iajb_S_kcld



    def compute_spatial_aats(self, alpha, beta, normalization='full'):
        """
        Compute the atomic axial tensors from the spatial orbital basis.
        """
        t0 = time.time()
        # Compute normalization factors.
        if self.parameters['method'] == 'RHF' or normalization == 'intermediate':
            N = 1 
            N_np = 1 
            N_nn = 1 
            N_mp = 1 
            N_mn = 1
        elif normalization == 'full' and self.parameters['method'] != 'CISD':
            N = 1 / np.sqrt(self.unperturbed_T[0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]) - np.einsum('ijab,ijba->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2])))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2])))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2])))
            N_mp = 1 / np.sqrt(self.mag_pos_T[beta][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2])))
            N_mn = 1 / np.sqrt(self.mag_neg_T[beta][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2])))
        elif normalization == 'full' and self.parameters['method'] == 'CISD':
            N = 1 / np.sqrt(self.unperturbed_T[0] + 2*np.einsum('ia,ia->', np.conjugate(self.unperturbed_T[1]), self.unperturbed_T[1]) + (2*np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]) - np.einsum('ijab,ijba->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2])))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0] + 2*np.einsum('ia,ia->', np.conjugate(self.nuc_pos_T[alpha][1]), self.nuc_pos_T[alpha][1]) + (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2])))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0] + 2*np.einsum('ia,ia->', np.conjugate(self.nuc_neg_T[alpha][1]), self.nuc_neg_T[alpha][1]) + (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2])))
            N_mp = 1 / np.sqrt(self.mag_pos_T[beta][0] + 2*np.einsum('ia,ia->', np.conjugate(self.mag_pos_T[beta][1]), self.mag_pos_T[beta][1]) + (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2])))
            N_mn = 1 / np.sqrt(self.mag_neg_T[beta][0] + 2*np.einsum('ia,ia->', np.conjugate(self.mag_neg_T[beta][1]), self.mag_neg_T[beta][1]) + (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2])))

        # Compute the HF AAT.
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
                # t_ia
                t1 = N * self.unperturbed_T[1]

                # dt_ia / dH
                t1_dH = N_mp * self.mag_pos_T[beta][1] - N_mn * self.mag_neg_T[beta][1]

                # t_ia*
                t1_conj = np.conjugate(t1)

                # dt_ijab / dR
                t1_dR = np.conjugate(N_np * self.nuc_pos_T[alpha][1] - N_nn * self.nuc_neg_T[alpha][1])

            # t_ijab
            t2 = N * self.unperturbed_T[2]

            # dt_ijab / dH
            t2_dH = N_mp * self.mag_pos_T[beta][2] - N_mn * self.mag_neg_T[beta][2]

            # t_ijab*
            t2_conj = np.conjugate(t2)

            # dt_ijab / dR
            t2_dR = np.conjugate(N_np * self.nuc_pos_T[alpha][2] - N_nn * self.nuc_neg_T[alpha][2])

            # Unperturbed / Unperturbed
            S_uu, ia_S_uu, S_kc_uu, iajb_S_uu, S_kcld_uu, ia_S_kc_uu, iajb_S_kc_uu, ia_S_kcld_uu, iajb_S_kcld_uu = self.compute_all_dets(self.overlap_uu)

            if self.parameters['method'] == 'CISD':
                # dt_ia/dR dt_kc/dH < ia | kc >
                I_SS += 2 * np.einsum('ia,kc,iakc->', t1_dR, t1_dH, ia_S_kc_uu) * S_uu
                I_SS += 2 * np.einsum('ia,ia->', t1_dR, ia_S_uu) * np.einsum('kc,kc->', t1_dH, S_kc_uu)

                # dt_ijab/dR dt_kc/dH < ijab | kc >
                I_DS += 0.5 * np.einsum('ijab,kc,iajbkc->', t2_dR - np.swapaxes(t2_dR, 2, 3), t1_dH, iajb_S_kc_uu) * S_uu
                I_DS += 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_uu) * np.einsum('kc,kc->', t1_dH, S_kc_uu)
                I_DS += 2 * np.einsum('ijab,kc,iakc,jb->', t2_dR, t1_dH, ia_S_kc_uu, ia_S_uu)

                # dt_ia/dR dt_klcd/dH < ia | klcd >
                I_SD += 0.5 * np.einsum('ia,klcd,iakcld->', t1_dR, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_uu) * S_uu
                I_SD += 0.5 * np.einsum('ia,ia->', t1_dR, ia_S_uu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_uu)
                I_SD += 2 * np.einsum('ia,klcd,iakc,ld->', t1_dR, t2_dH, ia_S_kc_uu, S_kc_uu)

            # dt_ijab/dR dt_klcd/dH < ijab | klcd >
            I_DD += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2_dH - np.swapaxes(t2_dH, 2, 3), iajb_S_kcld_uu) * S_uu
            I_DD += 0.125 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_uu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_uu)
            I_DD += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2_dH, iajb_S_kc_uu, S_kc_uu)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_dR, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_uu, ia_S_uu)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_dR, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_uu, ia_S_kcld_uu)
            I_DD += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_dR, t2_dH, ia_S_kc_uu, ia_S_kc_uu)
            del S_uu; del ia_S_uu; del S_kc_uu; del iajb_S_uu; del S_kcld_uu; del ia_S_kc_uu; del iajb_S_kc_uu; del ia_S_kcld_uu; del iajb_S_kcld_uu
            gc.collect()

            # Unperturbed / Positive
            S_up, ia_S_up, S_kc_up, iajb_S_up, S_kcld_up, ia_S_kc_up, iajb_S_kc_up, ia_S_kcld_up, iajb_S_kcld_up = self.compute_all_dets(self.overlap_up[beta])

            if self.parameters['method'] == 'CISD':
                # dt_ia/dR < ia | d0/dH >
                I_S0 += 2 * np.einsum('ia,ia->', t1_dR, ia_S_up) * S_up * N_mp

                # dt_ijab/dR < ijab | d0/dH >
                I_D0 += 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * S_up
                I_D0 += np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_dR, ia_S_up), ia_S_up)

                # dt_ia/dR t_kc < ia | dkc/dH >
                I_SS += 2 * np.einsum('ia,kc,iakc->', t1_dR, t1, ia_S_kc_up) * S_up
                I_SS += 2 * np.einsum('ia,ia->', t1_dR, ia_S_up) * np.einsum('kc,kc->', t1, S_kc_up)

                # dt_ijab/dR t_kc < ijab | dkc/dH >
                I_DS += 0.5 * np.einsum('ijab,kc,iajbkc->', t2_dR - np.swapaxes(t2_dR, 2, 3), t1, iajb_S_kc_up) * S_up
                I_DS += 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * np.einsum('kc,kc->', t1, S_kc_up)
                I_DS += 2 * np.einsum('ijab,kc,iakc,jb->', t2_dR, t1, ia_S_kc_up, ia_S_up)

                # dt_ia/dR t_klcd < ia | dklcd/dH >
                I_SD += 0.5 * np.einsum('ia,klcd,iakcld->', t1_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_up) * S_up
                I_SD += 0.5 * np.einsum('ia,ia->', t1_dR, ia_S_up) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_up)
                I_SD += 2 * np.einsum('ia,klcd,iakc,ld->', t1_dR, t2, ia_S_kc_up, S_kc_up)

            # dt_ijab/dR t_klcd < ijab | dklcd/dH >
            I_DD += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_up) * S_up
            I_DD += 0.125 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_up)
            I_DD += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2, iajb_S_kc_up, S_kc_up)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_up, ia_S_up)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_up, ia_S_kcld_up)
            I_DD += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_dR, t2, ia_S_kc_up, ia_S_kc_up)
            del S_up; del ia_S_up; del S_kc_up; del iajb_S_up; del S_kcld_up; del ia_S_kc_up; del iajb_S_kc_up; del ia_S_kcld_up; del iajb_S_kcld_up
            gc.collect()

            # Unperturbed / Negative
            S_un, ia_S_un, S_kc_un, iajb_S_un, S_kcld_un, ia_S_kc_un, iajb_S_kc_un, ia_S_kcld_un, iajb_S_kcld_un = self.compute_all_dets(self.overlap_un[beta])

            if self.parameters['method'] == 'CISD':
                # dt_ia/dR < ia | d0/dH >
                I_S0 -= 2 * np.einsum('ia,ia->', t1_dR, ia_S_un) * S_un * N_mn

                # dt_ijab/dR < ijab | d0/dH >
                I_D0 -= 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * S_un
                I_D0 -= np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_dR, ia_S_un), ia_S_un)

                # dt_ia/dR t_kc < ia | dkc/dH >
                I_SS -= 2 * np.einsum('ia,kc,iakc->', t1_dR, t1, ia_S_kc_un) * S_un
                I_SS -= 2 * np.einsum('ia,ia->', t1_dR, ia_S_un) * np.einsum('kc,kc->', t1, S_kc_un)

                # dt_ijab/dR t_kc < ijab | dkc/dH >
                I_DS -= 0.5 * np.einsum('ijab,kc,iajbkc->', t2_dR - np.swapaxes(t2_dR, 2, 3), t1, iajb_S_kc_un) * S_un
                I_DS -= 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * np.einsum('kc,kc->', t1, S_kc_un)
                I_DS -= 2 * np.einsum('ijab,kc,iakc,jb->', t2_dR, t1, ia_S_kc_un, ia_S_un)

                # dt_ia/dR t_klcd < ia | dklcd/dH >
                I_SD -= 0.5 * np.einsum('ia,klcd,iakcld->', t1_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_un) * S_un
                I_SD -= 0.5 * np.einsum('ia,ia->', t1_dR, ia_S_un) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_un)
                I_SD -= 2 * np.einsum('ia,klcd,iakc,ld->', t1_dR, t2, ia_S_kc_un, S_kc_un)

            # dt_ijab/dR t_klcd < ijab | dklcd/dH >
            I_DD -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_un) * S_un
            I_DD -= 0.125 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_un)
            I_DD -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2, iajb_S_kc_un, S_kc_un)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_un, ia_S_un)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_un, ia_S_kcld_un)
            I_DD -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_dR, t2, ia_S_kc_un, ia_S_kc_un)
            del S_un; del ia_S_un; del S_kc_un; del iajb_S_un; del S_kcld_un; del ia_S_kc_un; del iajb_S_kc_un; del ia_S_kcld_un; del iajb_S_kcld_un
            gc.collect()

            # Positive / Unperturbed
            S_pu, ia_S_pu, S_kc_pu, iajb_S_pu, S_kcld_pu, ia_S_kc_pu, iajb_S_kc_pu, ia_S_kcld_pu, iajb_S_kcld_pu = self.compute_all_dets(self.overlap_pu[alpha])

            if self.parameters['method'] == 'CISD':
                # dt_kc/dH < d0/dR | kc >
                I_0S += 2 * np.einsum('kc,kc->', t1_dH, S_kc_pu) * S_pu * N_np

                # dt_ijab/dH < d0/dR | klcd >
                I_0D += 0.5 * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_pu) * S_pu
                I_0D += np.einsum('ld,ld->', np.einsum('klcd,kc->ld', t2_dH, S_kc_pu), S_kc_pu)

                # t_ia dt_kc/dH < dia/dR | kc >
                I_SS += 2 * np.einsum('ia,kc,iakc->', t1_conj, t1_dH, ia_S_kc_pu) * S_pu
                I_SS += 2 * np.einsum('ia,ia->', t1_conj, ia_S_pu) * np.einsum('kc,kc->', t1_dH, S_kc_pu)

                # t_ia dt_klcd/dH < dia/dR | klcd >
                I_SD += 0.5 * np.einsum('ia,klcd,iakcld->', t1_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_pu) * S_pu
                I_SD += 0.5 * np.einsum('ia,ia->', t1_conj, ia_S_pu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_pu)
                I_SD += 2 * np.einsum('ia,klcd,iakc,ld->', t1_conj, t2_dH, ia_S_kc_pu, S_kc_pu)

                # t_ijab dt_kc/dH < dijab/dR | kc >
                I_DS += 0.5 * np.einsum('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1_dH, iajb_S_kc_pu) * S_pu
                I_DS += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pu) * np.einsum('kc,kc->', t1_dH, S_kc_pu)
                I_DS += 2 * np.einsum('ijab,kc,iakc,jb->', t2_conj, t1_dH, ia_S_kc_pu, ia_S_pu)

            # t_ijab dt_klcd/dH < dijab/dR | klcd >
            I_DD += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH - np.swapaxes(t2_dH, 2, 3), iajb_S_kcld_pu) * S_pu
            I_DD += 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_pu)
            I_DD += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH, iajb_S_kc_pu, S_kc_pu)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_pu, ia_S_pu)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_pu, ia_S_kcld_pu)
            I_DD += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2_dH, ia_S_kc_pu, ia_S_kc_pu)
            del S_pu; del ia_S_pu; del S_kc_pu; del iajb_S_pu; del S_kcld_pu; del ia_S_kc_pu; del iajb_S_kc_pu; del ia_S_kcld_pu; del iajb_S_kcld_pu
            gc.collect()

            # Negative / Unperturbed
            S_nu, ia_S_nu, S_kc_nu, iajb_S_nu, S_kcld_nu, ia_S_kc_nu, iajb_S_kc_nu, ia_S_kcld_nu, iajb_S_kcld_nu = self.compute_all_dets(self.overlap_nu[alpha])

            if self.parameters['method'] == 'CISD':
                # dt_kc/dH < d0/dR | kc >
                I_0S -= 2 * np.einsum('kc,kc->', t1_dH, S_kc_nu) * S_nu * N_nn

                # dt_ijab/dH < d0/dR | klcd >
                I_0D -= 0.5 * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_nu) * S_nu
                I_0D -= np.einsum('ld,ld->', np.einsum('klcd,kc->ld', t2_dH, S_kc_nu), S_kc_nu)

                # t_ia dt_kc/dH < dia/dR | kc >
                I_SS -= 2 * np.einsum('ia,kc,iakc->', t1_conj, t1_dH, ia_S_kc_nu) * S_nu
                I_SS -= 2 * np.einsum('ia,ia->', t1_conj, ia_S_nu) * np.einsum('kc,kc->', t1_dH, S_kc_nu)

                # t_ia dt_klcd/dH < dia/dR | klcd >
                I_SD -= 0.5 * np.einsum('ia,klcd,iakcld->', t1_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_nu) * S_nu
                I_SD -= 0.5 * np.einsum('ia,ia->', t1_conj, ia_S_nu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_nu)
                I_SD -= 2 * np.einsum('ia,klcd,iakc,ld->', t1_conj, t2_dH, ia_S_kc_nu, S_kc_nu)

                # t_ijab dt_kc/dH < dijab/dR | kc >
                I_DS -= 0.5 * np.einsum('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1_dH, iajb_S_kc_nu) * S_nu
                I_DS -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nu) * np.einsum('kc,kc->', t1_dH, S_kc_nu)
                I_DS -= 2 * np.einsum('ijab,kc,iakc,jb->', t2_conj, t1_dH, ia_S_kc_nu, ia_S_nu)

            # t_ijab dt_klcd/dH < dijab/dR | klcd >
            I_DD -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH - np.swapaxes(t2_dH, 2, 3), iajb_S_kcld_nu) * S_nu
            I_DD -= 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_nu)
            I_DD -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH, iajb_S_kc_nu, S_kc_nu)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_nu, ia_S_nu)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_nu, ia_S_kcld_nu)
            I_DD -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2_dH, ia_S_kc_nu, ia_S_kc_nu)
            del S_nu; del ia_S_nu; del S_kc_nu; del iajb_S_nu; del S_kcld_nu; del ia_S_kc_nu; del iajb_S_kc_nu; del ia_S_kcld_nu; del iajb_S_kcld_nu
            gc.collect()

            # Positive / Positive
            S_pp, ia_S_pp, S_kc_pp, iajb_S_pp, S_kcld_pp, ia_S_kc_pp, iajb_S_kc_pp, ia_S_kcld_pp, iajb_S_kcld_pp = self.compute_all_dets(self.overlap_pp[alpha][beta])

            if self.parameters['method'] == 'CISD':
                # t_kc < d0/dR | dkc/dH >
                I_0S += 2 * np.einsum('kc,kc->', t1, S_kc_pp) * S_pp * N_np

                # t_ia < dia/dR | d0/dH >
                I_S0 += 2 * np.einsum('ia,ia->', t1_conj, ia_S_pp) * S_pp * N_mp

                # t_klcd < d0/dR | dklcd/dH >
                I_0D += 0.5 * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp) * S_pp
                I_0D += np.einsum('ld,ld->', np.einsum('klcd,kc->ld', t2, S_kc_pp), S_kc_pp)

                # t_ijab < dijab/dR | d0/dH >
                I_D0 += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * S_pp
                I_D0 += np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_pp), ia_S_pp)

                # t_ia t_kc < dia/dR | dkc/dH >
                I_SS += 2 * np.einsum('ia,kc,iakc->', t1_conj, t1, ia_S_kc_pp) * S_pp
                I_SS += 2 * np.einsum('ia,ia->', t1_conj, ia_S_pp) * np.einsum('kc,kc->', t1, S_kc_pp)

                # t_ia t_klcd < dia/dR | dklcd/dH >
                I_SD += 0.5 * np.einsum('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pp) * S_pp
                I_SD += 0.5 * np.einsum('ia,ia->', t1_conj, ia_S_pp) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp)
                I_SD += 2 * np.einsum('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_pp, S_kc_pp)

                # t_ijab t_kc < dijab/dR | dkc/dH >
                I_DS += 0.5 * np.einsum('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_pp) * S_pp
                I_DS += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * np.einsum('kc,kc->', t1, S_kc_pp)
                I_DS += 2 * np.einsum('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_pp, ia_S_pp)

            # t_ijab t_klcd < dijab/dR | dklcd/dH >
            I_DD += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_pp) * S_pp
            I_DD += 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp)
            I_DD += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_pp, S_kc_pp)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pp, ia_S_pp)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_pp, ia_S_kcld_pp)
            I_DD += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_pp, ia_S_kc_pp)
            del S_pp; del ia_S_pp; del S_kc_pp; del iajb_S_pp; del S_kcld_pp; del ia_S_kc_pp; del iajb_S_kc_pp; del ia_S_kcld_pp; del iajb_S_kcld_pp
            gc.collect()

            # Positive / Negative
            S_pn, ia_S_pn, S_kc_pn, iajb_S_pn, S_kcld_pn, ia_S_kc_pn, iajb_S_kc_pn, ia_S_kcld_pn, iajb_S_kcld_pn = self.compute_all_dets(self.overlap_pn[alpha][beta])

            if self.parameters['method'] == 'CISD':
                # t_kc < d0/dR | dkc/dH >
                I_0S -= 2 * np.einsum('kc,kc->', t1, S_kc_pn) * S_pn * N_np

                # t_ia < dia/dR | d0/dH >
                I_S0 -= 2 * np.einsum('ia,ia->', t1_conj, ia_S_pn) * S_pn * N_mn

                # t_klcd < d0/dR | dklcd/dH >
                I_0D -= 0.5 * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn) * S_pn
                I_0D -= np.einsum('ld,ld->', np.einsum('klcd,kc->ld', t2, S_kc_pn), S_kc_pn)

                # t_ijab < dijab/dR | d0/dH >
                I_D0 -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * S_pn
                I_D0 -= np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_pn), ia_S_pn)

                # t_ia t_kc < dia/dR | dkc/dH >
                I_SS -= 2 * np.einsum('ia,kc,iakc->', t1_conj, t1, ia_S_kc_pn) * S_pn
                I_SS -= 2 * np.einsum('ia,ia->', t1_conj, ia_S_pn) * np.einsum('kc,kc->', t1, S_kc_pn)

                # t_ia t_klcd < dia/dR | dklcd/dH >
                I_SD -= 0.5 * np.einsum('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pn) * S_pn
                I_SD -= 0.5 * np.einsum('ia,ia->', t1_conj, ia_S_pn) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn)
                I_SD -= 2 * np.einsum('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_pn, S_kc_pn)

                # t_ijab t_kc < dijab/dR | dkc/dH >
                I_DS -= 0.5 * np.einsum('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_pn) * S_pn
                I_DS -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * np.einsum('kc,kc->', t1, S_kc_pn)
                I_DS -= 2 * np.einsum('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_pn, ia_S_pn)

            # t_ijab t_klcd < dijab/dR | dklcd/dH >
            I_DD -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_pn) * S_pn
            I_DD -= 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn)
            I_DD -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_pn, S_kc_pn)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pn, ia_S_pn)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_pn, ia_S_kcld_pn)
            I_DD -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_pn, ia_S_kc_pn)
            del S_pn; del ia_S_pn; del S_kc_pn; del iajb_S_pn; del S_kcld_pn; del ia_S_kc_pn; del iajb_S_kc_pn; del ia_S_kcld_pn; del iajb_S_kcld_pn
            gc.collect()

            # Negative / Positive
            S_np, ia_S_np, S_kc_np, iajb_S_np, S_kcld_np, ia_S_kc_np, iajb_S_kc_np, ia_S_kcld_np, iajb_S_kcld_np = self.compute_all_dets(self.overlap_np[alpha][beta])

            if self.parameters['method'] == 'CISD':
                # t_kc < d0/dR | dkc/dH >
                I_0S -= 2 * np.einsum('kc,kc->', t1, S_kc_np) * S_np * N_nn

                # t_ia < dia/dR | d0/dH >
                I_S0 -= 2 * np.einsum('ia,ia->', t1_conj, ia_S_np) * S_np * N_mp

                # t_klcd < d0/dR | dklcd/dH >
                I_0D -= 0.5 * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np) * S_np
                I_0D -= np.einsum('ld,ld->', np.einsum('klcd,kc->ld', t2, S_kc_np), S_kc_np)

                # t_ijab < dijab/dR | d0/dH >
                I_D0 -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * S_np
                I_D0 -= np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_np), ia_S_np)

                # t_ia t_kc < dia/dR | dkc/dH >
                I_SS -= 2 * np.einsum('ia,kc,iakc->', t1_conj, t1, ia_S_kc_np) * S_np
                I_SS -= 2 * np.einsum('ia,ia->', t1_conj, ia_S_np) * np.einsum('kc,kc->', t1, S_kc_np)

                # t_ia t_klcd < dia/dR | dklcd/dH >
                I_SD -= 0.5 * np.einsum('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_np) * S_np
                I_SD -= 0.5 * np.einsum('ia,ia->', t1_conj, ia_S_np) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np)
                I_SD -= 2 * np.einsum('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_np, S_kc_np)

                # t_ijab t_kc < dijab/dR | dkc/dH >
                I_DS -= 0.5 * np.einsum('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_np) * S_np
                I_DS -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * np.einsum('kc,kc->', t1, S_kc_np)
                I_DS -= 2 * np.einsum('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_np, ia_S_np)

            # t_ijab t_klcd < dijab/dR | dklcd/dH >
            I_DD -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_np) * S_np
            I_DD -= 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np)
            I_DD -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_np, S_kc_np)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_np, ia_S_np)
            I_DD -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_np, ia_S_kcld_np)
            I_DD -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_np, ia_S_kc_np)
            del S_np; del ia_S_np; del S_kc_np; del iajb_S_np; del S_kcld_np; del ia_S_kc_np; del iajb_S_kc_np; del ia_S_kcld_np; del iajb_S_kcld_np
            gc.collect()

            # Negative / Negative
            S_nn, ia_S_nn, S_kc_nn, iajb_S_nn, S_kcld_nn, ia_S_kc_nn, iajb_S_kc_nn, ia_S_kcld_nn, iajb_S_kcld_nn = self.compute_all_dets(self.overlap_nn[alpha][beta])

            if self.parameters['method'] == 'CISD':
                # t_kc < d0/dR | dkc/dH >
                I_0S += 2 * np.einsum('kc,kc->', t1, S_kc_nn) * S_nn * N_nn

                # t_ia < dia/dR | d0/dH >
                I_S0 += 2 * np.einsum('ia,ia->', t1_conj, ia_S_nn) * S_nn * N_mn

                # t_klcd < d0/dR | dklcd/dH >
                I_0D += 0.5 * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn) * S_nn
                I_0D += np.einsum('ld,ld->', np.einsum('klcd,kc->ld', t2, S_kc_nn), S_kc_nn)

                # t_ijab < dijab/dR | d0/dH >
                I_D0 += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * S_nn
                I_D0 += np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_nn), ia_S_nn)

                # t_ia t_kc < dia/dR | dkc/dH >
                I_SS += 2 * np.einsum('ia,kc,iakc->', t1_conj, t1, ia_S_kc_nn) * S_nn
                I_SS += 2 * np.einsum('ia,ia->', t1_conj, ia_S_nn) * np.einsum('kc,kc->', t1, S_kc_nn)

                # t_ia t_klcd < dia/dR | dklcd/dH >
                I_SD += 0.5 * np.einsum('ia,klcd,iakcld->', t1_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_nn) * S_nn
                I_SD += 0.5 * np.einsum('ia,ia->', t1_conj, ia_S_nn) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn)
                I_SD += 2 * np.einsum('ia,klcd,iakc,ld->', t1_conj, t2, ia_S_kc_nn, S_kc_nn)

                # t_ijab t_kc < dijab/dR | dkc/dH >
                I_DS += 0.5 * np.einsum('ijab,kc,iajbkc->', t2_conj - np.swapaxes(t2_conj, 2, 3), t1, iajb_S_kc_nn) * S_nn
                I_DS += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * np.einsum('kc,kc->', t1, S_kc_nn)
                I_DS += 2 * np.einsum('ijab,kc,iakc,jb->', t2_conj, t1, ia_S_kc_nn, ia_S_nn)

            # t_ijab t_klcd < dijab/dR | dklcd/dH >
            I_DD += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_nn) * S_nn
            I_DD += 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn)
            I_DD += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_nn, S_kc_nn)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_nn, ia_S_nn)
            I_DD += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_nn, ia_S_kcld_nn)
            I_DD += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_nn, ia_S_kc_nn)
            del S_nn; del ia_S_nn; del S_kc_nn; del iajb_S_nn; del S_kcld_nn; del ia_S_kc_nn; del iajb_S_kc_nn; del ia_S_kcld_nn; del iajb_S_kcld_nn
            gc.collect()

            I = I_00 + I_0D + I_D0 + I_DD + I_0S + I_S0 + I_SS + I_SD + I_DS

            t1 = time.time()
            print(f"AAT element computed in {t1-t0} seconds.")

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag












