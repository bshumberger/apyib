"""Contains the class and functions associated with computing the rotational strength for VCD calculations by finite difference at the Hartree-Fock, MP2, CID, and CISD levels of theory."""

import psi4
import numpy as np
import math
import itertools as it
import time
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

        elif normalization == 'full':
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
        # Compute the reference/doubles determinants in the spin-orbital basis.
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

                                        # t_ijab
                                        t2 = self.unperturbed_T[2][k][l][c-nocc][d-nocc]

                                        I += 0.25**2 * t2_dR * t2_dH * (det_S_uu * N * N)
                                        I += 0.25**2 * t2_dR * t2 * (det_S_up * N * N_mp - det_S_un * N * N_mn)
                                        I += 0.25**2 * t2_conj * t2_dH * (det_S_pu * N_np * N - det_S_nu * N_nn * N)
                                        I += 0.25**2 * t2_conj * t2 * (det_S_pp * N_np * N_mp - det_S_pn * N_np * N_mn - det_S_np * N_nn * N_mp + det_S_nn * N_nn * N_mn)

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag



    def compute_SO_aats(self, alpha, beta, normalization='full'):
        # Compute the HF term of the AATs.
        I_00 = self.compute_SO_I_00(alpha, beta, normalization)
        # Compute the MP2 or CID contribution to the AATs.
        if self.parameters['method'] != 'RHF':
            I_0D = self.compute_SO_I_0D(alpha, beta, normalization)
            I_D0 = self.compute_SO_I_D0(alpha, beta, normalization)
            I_DD = self.compute_SO_I_DD(alpha, beta, normalization)
        else:
            I_0D = 0
            I_D0 = 0
            I_DD = 0

        I = I_00 + I_0D + I_D0 + I_DD

        return I



    def compute_all_dets(self, overlap):
        # Setting up occupied and virtual spaces.
        nf = self.nfzc
        no = self.ndocc
        nv = self.nbf - self.ndocc

        # Computes all the determinants required from the row and column swappings.
        S = np.copy(overlap)

        # Initialize nested lists for storing substituted overlap matrices.
        ia_S = [[np.copy(S) for _ in range(nv)] for _ in range(nf, no)]
        S_kc = [[np.copy(S) for _ in range(nv)] for _ in range(nf, no)]
        iajb_S = [[[[np.zeros_like(S) for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)]
        S_kcld = [[[[np.zeros_like(S) for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)]
        ia_S_kc = [[[[np.zeros_like(S) for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)]
        iajb_S_kc = [[[[[[np.zeros_like(S) for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)]
        ia_S_kcld = [[[[[[np.zeros_like(S) for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)] for _ in range(nv)] for _ in range(nf, no)]

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
                ia_S[i-nf][a][[i, a + no],:] = ia_S[i-nf][a][[a + no, i],:]
                S_kc[i-nf][a][:,[i, a + no]] = S_kc[i-nf][a][:,[a + no, i]]
                det_ia_S[i-nf][a] = np.linalg.det(ia_S[i-nf][a][0:no, 0:no])
                det_S_kc[i-nf][a] = np.linalg.det(S_kc[i-nf][a][0:no, 0:no])

                for j in range(i+1, no):
                    for b in range(a+1, nv):
                        #if j == i:
                        #    continue
                        #if b == a:
                        #    continue
                        iajb_S[i-nf][a][j-nf][b] = np.copy(ia_S[i-nf][a])
                        iajb_S[i-nf][a][j-nf][b][[j, b + no],:] = iajb_S[i-nf][a][j-nf][b][[b + no, j],:]
                        S_kcld[i-nf][a][j-nf][b] = np.copy(S_kc[i-nf][a])
                        S_kcld[i-nf][a][j-nf][b][:,[j, b + no]] = S_kcld[i-nf][a][j-nf][b][:,[b + no, j]]
                        det_iajb_S[i-nf][a][j-nf][b] = np.linalg.det(iajb_S[i-nf][a][j-nf][b][0:no, 0:no])
                        det_S_kcld[i-nf][a][j-nf][b] = np.linalg.det(S_kcld[i-nf][a][j-nf][b][0:no, 0:no])

                        for k in range(nf, no):
                            for c in range(0, nv):
                                iajb_S_kc[i-nf][a][j-nf][b][k-nf][c] = np.copy(iajb_S[i-nf][a][j-nf][b])
                                iajb_S_kc[i-nf][a][j-nf][b][k-nf][c][:,[k, c + no]] = iajb_S_kc[i-nf][a][j-nf][b][k-nf][c][:,[c + no, k]]
                                ia_S_kcld[i-nf][a][j-nf][b][k-nf][c] = np.copy(S_kcld[i-nf][a][j-nf][b])
                                ia_S_kcld[i-nf][a][j-nf][b][k-nf][c][[k, c + no],:] = ia_S_kcld[i-nf][a][j-nf][b][k-nf][c][[c + no, k],:]
                                det_iajb_S_kc[i-nf][a][j-nf][b][k-nf][c] = np.linalg.det(iajb_S_kc[i-nf][a][j-nf][b][k-nf][c][0:no, 0:no])
                                det_ia_S_kcld[i-nf][a][j-nf][b][k-nf][c] = np.linalg.det(ia_S_kcld[i-nf][a][j-nf][b][k-nf][c][0:no, 0:no])

                                for l in range(k+1, no):
                                    for d in range(c+1, nv):
                                        #if l == k:
                                        #    continue
                                        #if d == c:
                                        #    continue
                                        #iajb_S_kcld[i][a][j][b][k][c][l][d] = np.copy(iajb_S_kc[i][a][j][b][k][c])
                                        #iajb_S_kcld[i][a][j][b][k][c][l][d][:,[l, d + self.ndocc]] = iajb_S_kcld[i][a][j][b][k][c][l][d][:,[d + self.ndocc, l]]
                                        #det_iajb_S_kcld[i][a][j][b][k][c][l][d] = np.linalg.det(iajb_S_kcld[i][a][j][b][k][c][l][d][0:self.ndocc, 0:self.ndocc])
                                        iajb_S_kcld = np.copy(iajb_S_kc[i-nf][a][j-nf][b][k-nf][c])
                                        iajb_S_kcld[:,[l, d + no]] = iajb_S_kcld[:,[d + no, l]]
                                        det_iajb_S_kcld[i-nf][a][j-nf][b][k-nf][c][l-nf][d] = np.linalg.det(iajb_S_kcld[0:no, 0:no])


                for k in range(nf, no):
                    for c in range(0, nv):
                        ia_S_kc[i-nf][a][k-nf][c] = np.copy(ia_S[i-nf][a])
                        ia_S_kc[i-nf][a][k-nf][c][:,[k, c + no]] = ia_S_kc[i-nf][a][k-nf][c][:,[c + no, k]]
                        det_ia_S_kc[i-nf][a][k-nf][c] = np.linalg.det(ia_S_kc[i-nf][a][k-nf][c][0:no, 0:no])

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
        # Compute normalization factors.
        if self.parameters['method'] == 'RHF' or normalization == 'intermediate':
            N = 1 
            N_np = 1 
            N_nn = 1 
            N_mp = 1 
            N_mn = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(self.unperturbed_T[0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2]) - np.einsum('ijab,ijba->', np.conjugate(self.unperturbed_T[2]), self.unperturbed_T[2])))
            N_np = 1 / np.sqrt(self.nuc_pos_T[alpha][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_pos_T[alpha][2]), self.nuc_pos_T[alpha][2])))
            N_nn = 1 / np.sqrt(self.nuc_neg_T[alpha][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2]) - np.einsum('ijab,ijba->', np.conjugate(self.nuc_neg_T[alpha][2]), self.nuc_neg_T[alpha][2])))
            N_mp = 1 / np.sqrt(self.mag_pos_T[beta][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_pos_T[beta][2]), self.mag_pos_T[beta][2])))
            N_mn = 1 / np.sqrt(self.mag_neg_T[beta][0] + (2*np.einsum('ijab,ijab->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2]) - np.einsum('ijab,ijba->', np.conjugate(self.mag_neg_T[beta][2]), self.mag_neg_T[beta][2])))

        # Compute the HF AATs.
        S_pp = np.linalg.det(self.overlap_pp[alpha][beta][0:self.ndocc,0:self.ndocc])**2
        S_pn = np.linalg.det(self.overlap_pn[alpha][beta][0:self.ndocc,0:self.ndocc])**2
        S_np = np.linalg.det(self.overlap_np[alpha][beta][0:self.ndocc,0:self.ndocc])**2
        S_nn = np.linalg.det(self.overlap_nn[alpha][beta][0:self.ndocc,0:self.ndocc])**2

        I = S_pp * N_np * N_mp - S_pn * N_np * N_mn - S_np * N_nn * N_mp + S_nn * N_nn * N_mn

        if self.parameters['method'] != 'RHF':

            # Compute determinant lists.
            S_uu, ia_S_uu, S_kc_uu, iajb_S_uu, S_kcld_uu, ia_S_kc_uu, iajb_S_kc_uu, ia_S_kcld_uu, iajb_S_kcld_uu = self.compute_all_dets(self.overlap_uu)

            S_pu, ia_S_pu, S_kc_pu, iajb_S_pu, S_kcld_pu, ia_S_kc_pu, iajb_S_kc_pu, ia_S_kcld_pu, iajb_S_kcld_pu = self.compute_all_dets(self.overlap_pu[alpha])
            S_nu, ia_S_nu, S_kc_nu, iajb_S_nu, S_kcld_nu, ia_S_kc_nu, iajb_S_kc_nu, ia_S_kcld_nu, iajb_S_kcld_nu = self.compute_all_dets(self.overlap_nu[alpha])

            S_up, ia_S_up, S_kc_up, iajb_S_up, S_kcld_up, ia_S_kc_up, iajb_S_kc_up, ia_S_kcld_up, iajb_S_kcld_up = self.compute_all_dets(self.overlap_up[beta])
            S_un, ia_S_un, S_kc_un, iajb_S_un, S_kcld_un, ia_S_kc_un, iajb_S_kc_un, ia_S_kcld_un, iajb_S_kcld_un = self.compute_all_dets(self.overlap_un[beta])

            S_pp, ia_S_pp, S_kc_pp, iajb_S_pp, S_kcld_pp, ia_S_kc_pp, iajb_S_kc_pp, ia_S_kcld_pp, iajb_S_kcld_pp = self.compute_all_dets(self.overlap_pp[alpha][beta])
            S_pn, ia_S_pn, S_kc_pn, iajb_S_pn, S_kcld_pn, ia_S_kc_pn, iajb_S_kc_pn, ia_S_kcld_pn, iajb_S_kcld_pn = self.compute_all_dets(self.overlap_pn[alpha][beta])
            S_np, ia_S_np, S_kc_np, iajb_S_np, S_kcld_np, ia_S_kc_np, iajb_S_kc_np, ia_S_kcld_np, iajb_S_kcld_np = self.compute_all_dets(self.overlap_np[alpha][beta])
            S_nn, ia_S_nn, S_kc_nn, iajb_S_nn, S_kcld_nn, ia_S_kc_nn, iajb_S_kc_nn, ia_S_kcld_nn, iajb_S_kcld_nn = self.compute_all_dets(self.overlap_nn[alpha][beta])

            # t_ijab
            t2 = self.unperturbed_T[2]

            # dt_ijab / dH
            t2_dH = self.mag_pos_T[beta][2] - self.mag_neg_T[beta][2]

            # t_ijab*
            t2_conj = np.conjugate(t2)

            # dt_ijab / dR
            t2_dR = np.conjugate(self.nuc_pos_T[alpha][2] - self.nuc_neg_T[alpha][2])

            # < ijab | d0/dH >
            I += 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * S_up * N * N_mp
            I -= 0.5 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * S_un * N * N_mn

            I += 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_dR, ia_S_up), ia_S_up) * N * N_mp
            I -= 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_dR, ia_S_un), ia_S_un) * N * N_mn

            # < dijab/dR | d0/dH >
            I += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * S_pp * N_np * N_mp
            I -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * S_pn * N_np * N_mn
            I -= 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * S_np * N_nn * N_mp
            I += 0.5 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * S_nn * N_nn * N_mn

            I += 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_pp), ia_S_pp) * N_np * N_mp
            I -= 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_pn), ia_S_pn) * N_np * N_mn
            I -= 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_np), ia_S_np) * N_nn * N_mp
            I += 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_conj, ia_S_nn), ia_S_nn) * N_nn * N_mn

            # < d0/dR | ijab >
            I += 0.5 * np.einsum('ijab,iajb->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_pu) * S_pu * N_np * N
            I -= 0.5 * np.einsum('ijab,iajb->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_nu) * S_nu * N_nn * N

            I += 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_dR, ia_S_pu), ia_S_pu) * N_np * N
            I -= 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2_dR, ia_S_nu), ia_S_nu) * N_nn * N

            # < d0/dR | dijab/dH >
            I += 0.5 * np.einsum('ijab,iajb->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp) * S_pp * N_np * N_mp
            I -= 0.5 * np.einsum('ijab,iajb->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn) * S_pn * N_np * N_mn
            I -= 0.5 * np.einsum('ijab,iajb->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np) * S_np * N_nn * N_mp
            I += 0.5 * np.einsum('ijab,iajb->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn) * S_nn * N_nn * N_mn

            I += 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2, S_kc_pp), S_kc_pp) * N_np * N_mp
            I -= 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2, S_kc_pn), S_kc_pn) * N_np * N_mn
            I -= 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2, S_kc_np), S_kc_np) * N_nn * N_mp
            I += 0.5 * 2.0 * np.einsum('jb,jb->', np.einsum('ijab,ia->jb', t2, S_kc_nn), S_kc_nn) * N_nn * N_mn

            # < ijab | klcd >
            I += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2_dH - np.swapaxes(t2_dH, 2, 3), iajb_S_kcld_uu) * S_uu  * N * N
            I += 0.125 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_uu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_uu) * N * N
            I += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2_dH, iajb_S_kc_uu, S_kc_uu) * N * N

            I += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_dR, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_uu, ia_S_uu) * N * N
            I += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_dR, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_uu, ia_S_kcld_uu) * N * N
            I += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_dR, t2_dH, ia_S_kc_uu, ia_S_kc_uu) * N * N

            # < ijab | dklcd/dH >
            I += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_up) * S_up * N * N_mp
            I -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_un) * S_un * N * N_mn
            I += 0.125 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_up) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_up) * N * N_mp
            I -= 0.125 * np.einsum('ijab,iajb->', t2_dR - np.swapaxes(t2_dR, 2, 3), iajb_S_un) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_un) * N * N_mn
            I += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2, iajb_S_kc_up, S_kc_up) * N * N_mp
            I -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_dR - np.swapaxes(t2_dR, 2, 3), t2, iajb_S_kc_un, S_kc_un) * N * N_mn

            I += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_up, ia_S_up) * N * N_mp
            I -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_un, ia_S_un) * N * N_mn
            I += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_up, ia_S_kcld_up) * N * N_mp
            I -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_dR, t2 - np.swapaxes(t2, 2, 3), ia_S_un, ia_S_kcld_un) * N * N_mn
            I += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_dR, t2, ia_S_kc_up, ia_S_kc_up) * N * N_mp
            I -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_dR, t2, ia_S_kc_un, ia_S_kc_un) * N * N_mn

            # < dijab/dR | klcd >
            I += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH - np.swapaxes(t2_dH, 2, 3), iajb_S_kcld_pu) * S_pu * N_np * N
            I -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH - np.swapaxes(t2_dH, 2, 3), iajb_S_kcld_nu) * S_nu * N_nn * N
            I += 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_pu) * N_np * N
            I -= 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nu) * np.einsum('klcd,kcld->', t2_dH - np.swapaxes(t2_dH, 2, 3), S_kcld_nu) * N_nn * N
            I += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH, iajb_S_kc_pu, S_kc_pu) * N_np * N
            I -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2_dH, iajb_S_kc_nu, S_kc_nu) * N_nn * N

            I += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_pu, ia_S_pu) * N_np * N
            I -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_kcld_nu, ia_S_nu) * N_nn * N
            I += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_pu, ia_S_kcld_pu) * N_np * N
            I -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2_dH - np.swapaxes(t2_dH, 2, 3), ia_S_nu, ia_S_kcld_nu) * N_nn * N
            I += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2_dH, ia_S_kc_pu, ia_S_kc_pu) * N_np * N
            I -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2_dH, ia_S_kc_nu, ia_S_kc_nu) * N_nn * N

            # < dijab/dR | dklcd/dH >
            I += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_pp) * S_pp * N_np * N_mp
            I -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_pn) * S_pn * N_np * N_mn
            I -= 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_np) * S_np * N_nn * N_mp
            I += 0.125 * np.einsum('ijab,klcd,iajbkcld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2 - np.swapaxes(t2, 2, 3), iajb_S_kcld_nn) * S_nn * N_nn * N_mn

            I += 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pp) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pp) * N_np * N_mp
            I -= 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_pn) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_pn) * N_np * N_mn
            I -= 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_np) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_np) * N_nn * N_mp
            I += 0.125 * np.einsum('ijab,iajb->', t2_conj - np.swapaxes(t2_conj, 2, 3), iajb_S_nn) * np.einsum('klcd,kcld->', t2 - np.swapaxes(t2, 2, 3), S_kcld_nn) * N_nn * N_mn

            I += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_pp, S_kc_pp) * N_np * N_mp
            I -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_pn, S_kc_pn) * N_np * N_mn
            I -= 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_np, S_kc_np) * N_nn * N_mp
            I += 0.125 * 4 * np.einsum('ijab,klcd,iajbkc,ld->', t2_conj - np.swapaxes(t2_conj, 2, 3), t2, iajb_S_kc_nn, S_kc_nn) * N_nn * N_mn

            I += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pp, ia_S_pp) * N_np * N_mp
            I -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_pn, ia_S_pn) * N_np * N_mn
            I -= 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_np, ia_S_np) * N_nn * N_mp
            I += 0.125 * 2 * np.einsum('ijab,klcd,iakcld,jb->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_kcld_nn, ia_S_nn) * N_nn * N_mn

            I += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_pp, ia_S_kcld_pp) * N_np * N_mp
            I -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_pn, ia_S_kcld_pn) * N_np * N_mn
            I -= 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_np, ia_S_kcld_np) * N_nn * N_mp
            I += 0.125 * 2 * np.einsum('ijab,klcd,ia,jbkcld->', t2_conj, t2 - np.swapaxes(t2, 2, 3), ia_S_nn, ia_S_kcld_nn) * N_nn * N_mn

            I += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_pp, ia_S_kc_pp) * N_np * N_mp
            I -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_pn, ia_S_kc_pn) * N_np * N_mn
            I -= 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_np, ia_S_kc_np) * N_nn * N_mp
            I += 0.125 * 2 * 4 * np.einsum('ijab,klcd,iakc,jbld->', t2_conj, t2, ia_S_kc_nn, ia_S_kc_nn) * N_nn * N_mn

        return (1 / (4 * self.nuc_pert_strength * self.mag_pert_strength)) * I.imag












