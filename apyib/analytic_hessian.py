"""This script contains a set of functions for analytic evaluation of the Hessian."""

import numpy as np
import psi4
import gc
import opt_einsum as oe
from apyib.analytic_base import AnalyticDerivative
from apyib.utils import compute_ERI_MO
from apyib.utils import solve_general_DIIS


class analytic_derivative(AnalyticDerivative):
    """Analytic nuclear Hessian for RHF wavefunctions."""

    def compute_RHF_Hessian(self, orbitals='non-canonical'):
        """Compute the analytic RHF nuclear Hessian.

        Parameters
        ----------
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention.  ``'non-canonical'`` (default) uses
            block-diagonal MOs; ``'canonical'`` uses fully diagonalised MOs.

        Returns
        -------
        Hessian : ndarray, shape (3*natom, 3*natom)
            Second derivative of the total energy w.r.t. nuclear coordinates
            [E_h / a_0^2].
        """
        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints

        N = self.H.molecule.nuclear_repulsion_energy_deriv2().np
        Hessian = np.zeros((natom * 3, natom * 3))

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        npert = 3 * natom
        B_nuc = np.zeros((nv * no, npert))
        S_nuc = np.zeros((npert, nbf, nbf))
        h_dep_nuc = np.zeros((npert, nbf, nbf))
        h_R = []
        ERI_R = []
        S_R = []
        F_R = []

        # First pass: collect all first-derivative integrals and build CPHF B vectors.
        for N1 in atoms:
            T_a = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_a = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_a = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_a = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            for a in range(3):
                T_a[a] = T_a[a].np
                V_a[a] = V_a[a].np
                S_a[a] = S_a[a].np
                ERI_a[a] = ERI_a[a].np
                ERI_a[a] = ERI_a[a].swapaxes(1, 2)

                # Skeleton-derivative Fock matrix for this Cartesian displacement.
                h_a = T_a[a] + V_a[a]
                F_a = h_a + oe.contract('piqi->pq', 2 * ERI_a[a][:, o, :, o] - ERI_a[a].swapaxes(2, 3)[:, o, :, o])

                # CPHF right-hand side for nuclear perturbation k = 3*N1 + a:
                # skeleton Fock term plus overlap-derivative corrections.
                k = 3 * N1 + a
                B_nuc[:, k] = (-F_a[v, o]
                                + oe.contract('ai,ii->ai', S_a[a][v, o], F[o, o])
                                + 0.5 * oe.contract('mn,amin->ai', S_a[a][o, o], A.swapaxes(1, 2)[v, o, o, o])
                                ).reshape(nv * no)
                S_nuc[k] = S_a[a]
                h_dep_nuc[k] = F_a
                h_R.append(h_a)
                ERI_R.append(ERI_a[a])
                S_R.append(S_a[a])
                F_R.append(F_a)

        # Vectorized nuclear CPHF solve for all 3*natom directions simultaneously.
        # ov_sign=-1: real perturbation (U[o,v] = -U[v,o].T - S[o,v]).
        # dep_sign=+1: B_oo = +F_a[o,o] - S*F + U*A - 0.5*S*A.
        U_R = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        # Second pass: Hessian contractions using stored first-derivative data and solved U matrices.
        for N1 in atoms:
            for N2 in atoms:
                T_ab = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_ab = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_ab = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)
                ERI_ab = mints.mo_tei_deriv2(N1, N2, C_p4, C_p4, C_p4, C_p4)

                for a in range(3):
                    for b in range(3):
                        ab = 3 * a + b
                        N1a = 3 * N1 + a
                        N2b = 3 * N2 + b
                        T_RR = T_ab[ab].np
                        V_RR = V_ab[ab].np
                        S_RR = S_ab[ab].np
                        h_RR = T_RR + V_RR
                        ERI_RR = ERI_ab[ab].np
                        ERI_RR = ERI_RR.swapaxes(1, 2)

                        # Hessian element = explicit second-derivative integral terms
                        # (h_RR, ERI_RR) plus orbital-response (U_R) and overlap-
                        # derivative (S_R, S_RR) coupling terms.
                        Hessian[N1a][N2b] += 2 * oe.contract('ii->', h_RR[o, o])
                        Hessian[N1a][N2b] += 1 * oe.contract('ijij->', 2 * ERI_RR[o, o, o, o] - ERI_RR[o, o, o, o].swapaxes(2, 3))
                        Hessian[N1a][N2b] += 2 * oe.contract('pi,pi->', U_R[N2b][:, o], F_R[N1a][:, o] + F_R[N1a][o, :].T)
                        Hessian[N1a][N2b] -= 2 * oe.contract('pi,pj,ij->', U_R[N2b][:, o], S_R[N1a][:, o], F[o, o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('pj,ip,ij->', U_R[N2b][:, o], S_R[N1a][o, :], F[o, o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,ij->', S_RR[o, o], F[o, o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,ij->', S_R[N1a][o, o], F_R[N2b][o, o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,ki,kj->', S_R[N1a][o, o], U_R[N2b][o, o], F[o, o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,kj,ik->', S_R[N1a][o, o], U_R[N2b][o, o], F[o, o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,pk,ipjk->', S_R[N1a][o, o], U_R[N2b][:, o],
                                               2 * ERI[o, :, o, o] + 2 * ERI[o, o, o, :].swapaxes(1, 3)
                                               - ERI[o, :, o, o].swapaxes(2, 3)
                                               - ERI[o, o, :, o].swapaxes(1, 2).swapaxes(2, 3))

        Hessian += N
        return Hessian



    def compute_RHF_Hessian_opt(self, orbitals='non-canonical'):
        """Compute the analytic RHF nuclear Hessian using frozen-core-aware ERIs.

        Identical result to :meth:`compute_RHF_Hessian` but uses a frozen-core
        ERI transform; intended for benchmarking frozen-core energetics.

        Parameters
        ----------
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).

        Returns
        -------
        Hessian : ndarray, shape (3*natom, 3*natom)
            Second derivative of the total energy w.r.t. nuclear coordinates
            [E_h / a_0^2].
        """
        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, mints = m.h, m.mints

        # Use the frozen-core-aware ERI transform for the optimised variant.
        ERI = compute_ERI_MO(self.parameters, self.wfn, m.C_list)
        ERI = ERI.swapaxes(1, 2)  # (pr|qs) -> <pq|rs>
        F = h + oe.contract('piqi->pq', 2 * ERI[:, o, :, o] - ERI.swapaxes(2, 3)[:, o, :, o])

        N = self.H.molecule.nuclear_repulsion_energy_deriv2().np
        Hessian = np.zeros((natom * 3, natom * 3))

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        npert = 3 * natom
        B_nuc = np.zeros((nv * no, npert))
        S_nuc = np.zeros((npert, nbf, nbf))
        h_dep_nuc = np.zeros((npert, nbf, nbf))
        h_Ra_store = []
        F_Ra_store = []
        S_Ra_store = []

        # First pass: collect all first-derivative integrals and CPHF B vectors.
        for N1 in atoms:
            T_Ra = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_Ra = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_Ra = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_Ra = mints.ao_tei_deriv1(N1)

            for a in range(3):
                T_Ra[a] = T_Ra[a].np
                V_Ra[a] = V_Ra[a].np
                S_Ra[a] = S_Ra[a].np
                ERI_Ra[a] = ERI_Ra[a].np
                ERI_Ra[a] = oe.contract('mnlg,gs->mnls', ERI_Ra[a], C)
                ERI_Ra[a] = oe.contract('mnls,lr->mnrs', ERI_Ra[a], np.conjugate(C))
                ERI_Ra[a] = oe.contract('nq,mnrs->mqrs', C, ERI_Ra[a])
                ERI_Ra[a] = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI_Ra[a])
                ERI_Ra[a] = ERI_Ra[a].swapaxes(1, 2)

                h_Ra = T_Ra[a] + V_Ra[a]
                F_Ra = h_Ra + oe.contract('piqi->pq', 2 * ERI_Ra[a][:, o, :, o] - ERI_Ra[a].swapaxes(2, 3)[:, o, :, o])

                k = 3 * N1 + a
                B_nuc[:, k] = (-F_Ra[v, o]
                                + oe.contract('ai,ii->ai', S_Ra[a][v, o], F[o, o])
                                + 0.5 * oe.contract('mn,amin->ai', S_Ra[a][o, o], A.swapaxes(1, 2)[v, o, o, o])
                                ).reshape(nv * no)
                S_nuc[k] = S_Ra[a]
                h_dep_nuc[k] = F_Ra
                h_Ra_store.append(h_Ra)
                F_Ra_store.append(F_Ra)
                S_Ra_store.append(S_Ra[a])

        # Vectorized nuclear CPHF solve for all 3*natom directions simultaneously.
        # ov_sign=-1: real perturbation; S_all provides overlap derivatives.
        # dep_sign=+1: B_oo = +F_a[o,o] - S*F + U*A - 0.5*S*A.
        U_R = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        # Second pass: symmetric Hessian contractions (upper triangle only, then symmetrize).
        for N1 in atoms:
            for N2 in range(N1, natom):
                T_RaRb = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_RaRb = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_RaRb = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)
                ERI_RaRb = mints.ao_tei_deriv2(N1, N2)

                for a in range(3):
                    for b in range(3):
                        ab = 3 * a + b
                        N1a = 3 * N1 + a
                        N2b = 3 * N2 + b

                        if N2b < N1a:
                            continue

                        T_RaRb[ab] = T_RaRb[ab].np
                        V_RaRb[ab] = V_RaRb[ab].np
                        S_RaRb[ab] = S_RaRb[ab].np
                        ERI_RaRb[ab] = ERI_RaRb[ab].np
                        ERI_RaRb[ab] = oe.contract('mnlg,gs->mnls', ERI_RaRb[ab], C)
                        ERI_RaRb[ab] = oe.contract('mnls,lr->mnrs', ERI_RaRb[ab], np.conjugate(C))
                        ERI_RaRb[ab] = oe.contract('nq,mnrs->mqrs', C, ERI_RaRb[ab])
                        ERI_RaRb[ab] = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI_RaRb[ab])
                        ERI_RaRb[ab] = ERI_RaRb[ab].swapaxes(1, 2)

                        h_RaRb = T_RaRb[ab] + V_RaRb[ab]

                        U_Ra = U_R[N1a]
                        U_Rb = U_R[N2b]
                        F_Ra = F_Ra_store[N1a]
                        F_Rb = F_Ra_store[N2b]
                        S_Ra_a = S_Ra_store[N1a]
                        S_Rb_b = S_Ra_store[N2b]

                        # Occupied-block second-order response intermediate:
                        # symmetrized U-matrix products minus overlap-derivative products.
                        eta_RR = (oe.contract('im,jm->ij', U_Ra[o, :], U_Rb[o, :])
                                  + oe.contract('im,jm->ij', U_Rb[o, :], U_Ra[o, :])
                                  - oe.contract('im,jm->ij', S_Ra_a[o, :], S_Rb_b[o, :])
                                  - oe.contract('im,jm->ij', S_Rb_b[o, :], S_Ra_a[o, :]))

                        Hessian[N1a][N2b] += 2 * oe.contract('ii->', h_RaRb[o, o])
                        Hessian[N1a][N2b] += 1 * oe.contract('ijij->', 2 * ERI_RaRb[ab][o, o, o, o] - ERI_RaRb[ab][o, o, o, o].swapaxes(2, 3))
                        Hessian[N1a][N2b] -= 2 * oe.contract('ii,i->', S_RaRb[ab][o, o], self.wfn.eps[o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ii,i->', eta_RR[o, o], self.wfn.eps[o])
                        Hessian[N1a][N2b] += 4 * oe.contract('ij,ij->', U_Ra[:, o], F_Rb[:, o]) + 4 * oe.contract('ij,ij->', U_Rb[:, o], F_Ra[:, o])
                        Hessian[N1a][N2b] += 4 * oe.contract('ij,ij,i->', U_Ra[:, o], U_Rb[:, o], self.wfn.eps[:])
                        Hessian[N1a][N2b] += 4 * oe.contract('ij,kl,ikjl->', U_Ra[:, o], U_Rb[:, o],
                                               4 * ERI[:, :, o, o] - ERI[:, :, o, o].swapaxes(2, 3)
                                               - ERI[:, o, :, o].swapaxes(1, 2).swapaxes(2, 3))

        Hessian += Hessian.T
        Hessian -= 0.5 * np.eye(3 * natom) * Hessian
        Hessian += N
        return Hessian
















