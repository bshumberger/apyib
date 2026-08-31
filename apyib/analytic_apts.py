"""This script contains a set of functions for analytic evaluation of the atomic polar tensors."""

import numpy as np
import opt_einsum as oe
from apyib.analytic_base import AnalyticDerivative
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn
from apyib.utils import solve_general_DIIS


class analytic_derivative(AnalyticDerivative):
    """Analytic atomic polar tensors for RHF wavefunctions."""



    def compute_RHF_APTs_LG(self, orbitals='non-canonical'):
        """Compute analytic RHF atomic polar tensors in the length gauge.

        Parameters
        ----------
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).

        Returns
        -------
        APT : ndarray, shape (3*natom, 3)
            Atomic polar tensor ``dmu_beta / dR_alpha`` [e].
        """
        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints

        APT = np.zeros((natom * 3, 3))

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        # Collect electric dipole integrals and build CPHF B vectors for 3 field directions.
        mu_AO = mints.ao_dipole()
        B_elec = np.zeros((nv * no, 3))
        h_dep_elec = np.zeros((3, nbf, nbf))
        h_E = []
        for b in range(3):
            mu_AO[b] = mu_AO[b].np
            mu = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_AO[b], C)
            # Electric-field CPHF right-hand side is just the (negative) MO dipole;
            # no skeleton-Fock or overlap-derivative terms (AOs are field-independent).
            B_elec[:, b] = -mu[v, o].reshape(nv * no)
            h_dep_elec[b] = mu
            h_E.append(mu)

        # Vectorized CPHF solve for all 3 electric field directions simultaneously.
        # ov_sign=-1: real perturbation (U[o,v] = -U[v,o].T, no overlap derivative).
        # dep_sign=+1: B_oo = +h[o,o] + U*A.
        U_E = self._solve_cphf(G, A, B_elec, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=-1, h_dep_all=h_dep_elec, dep_sign=+1)

        for N1 in atoms:
            T_a = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_a = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_a = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_a = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            h_ab = mints.ao_elec_dip_deriv1(N1)

            for a in range(3):
                T_R = T_a[a].np
                V_R = V_a[a].np
                S_R = S_a[a].np
                ERI_R = ERI_a[a].np
                ERI_R = ERI_R.swapaxes(1, 2)

                h_R = T_R + V_R
                F_R = h_R + oe.contract('pkqk->pq', 2 * ERI_R[:, o, :, o] - ERI_R[:, o, o, :].swapaxes(2, 3))

                lambda_alpha = 3 * N1 + a
                for beta in range(3):
                    h_RE = h_ab[a + 3 * beta].np
                    h_RE = oe.contract('mp,mn,nq->pq', np.conjugate(C), h_RE, C)

                    # APT element dmu_beta/dR_alpha: explicit mixed nuclear-field
                    # derivative (h_RE) plus orbital-response and overlap-derivative terms.
                    APT[lambda_alpha][beta] += 2 * oe.contract('ii->', h_RE[o, o])
                    APT[lambda_alpha][beta] += 2 * oe.contract('pi,pi->', U_E[beta][:, o], F_R[:, o] + F_R[o, :].T)
                    APT[lambda_alpha][beta] -= 2 * oe.contract('pi,pj,ij->', U_E[beta][:, o], S_R[:, o], F[o, o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('pj,ip,ij->', U_E[beta][:, o], S_R[o, :], F[o, o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,ij->', S_R[o, o], h_E[beta][o, o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,ki,kj->', S_R[o, o], U_E[beta][o, o], F[o, o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,kj,ik->', S_R[o, o], U_E[beta][o, o], F[o, o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,pk,ipjk->', S_R[o, o], U_E[beta][:, o],
                                               2 * ERI[o, :, o, o] + 2 * ERI[o, o, o, :].swapaxes(1, 3)
                                               - ERI[o, :, o, o].swapaxes(2, 3)
                                               - ERI[o, o, :, o].swapaxes(1, 2).swapaxes(2, 3))

        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()
        N = np.zeros((3 * natom, 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * natom):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                N[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        return APT + N



    def compute_RHF_APTs_VG(self, orbitals='non-canonical'):
        """Compute analytic RHF atomic polar tensors in the velocity gauge.

        Parameters
        ----------
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).

        Returns
        -------
        APT : ndarray, shape (3*natom, 3)
            Velocity-gauge atomic polar tensor ``-i * dp_beta / dR_alpha``
            [e * a_0^{-1}].
        """
        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        # Collect nuclear derivative integrals and build CPHF B vectors for all 3*natom directions.
        B_nuc = np.zeros((nv * no, 3 * natom))
        S_nuc = np.zeros((3 * natom, nbf, nbf))
        h_dep_nuc = np.zeros((3 * natom, nbf, nbf))
        half_S = []

        for N1 in atoms:
            T_d1 = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_d1 = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_d1 = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_d1 = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            half_S_d1 = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                T_d1[a] = T_d1[a].np
                V_d1[a] = V_d1[a].np
                S_d1[a] = S_d1[a].np
                ERI_d1[a] = ERI_d1[a].np
                ERI_d1[a] = ERI_d1[a].swapaxes(1, 2)
                half_S_d1[a] = half_S_d1[a].np

                h_d1 = T_d1[a] + V_d1[a]
                F_d1 = h_d1 + oe.contract('piqi->pq', 2 * ERI_d1[a][:, o, :, o] - ERI_d1[a].swapaxes(2, 3)[:, o, :, o])

                k = 3 * N1 + a
                B_nuc[:, k] = (-F_d1[v, o]
                                + oe.contract('ai,ii->ai', S_d1[a][v, o], F[o, o])
                                + 0.5 * oe.contract('mn,amin->ai', S_d1[a][o, o], A.swapaxes(1, 2)[v, o, o, o])
                                ).reshape(nv * no)
                S_nuc[k] = S_d1[a]
                h_dep_nuc[k] = F_d1
                half_S.append(half_S_d1[a])

        # Vectorized nuclear CPHF solve for all 3*natom directions simultaneously.
        # ov_sign=-1: real perturbation; S_all provides overlap derivatives.
        # dep_sign=+1: B_oo = +F_a[o,o] - S*F + U*A - 0.5*S*A.
        U_R = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        A_elec, G_elec = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        # Collect nabla (velocity-gauge) integrals and build CPHF B vectors for 3 field directions.
        mu_elec_AO = mints.ao_nabla()
        B_elec = np.zeros((nv * no, 3))
        h_dep_elec = np.zeros((3, nbf, nbf))
        for a in range(3):
            mu_elec_AO[a] = -mu_elec_AO[a].np
            mu_elec = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_elec_AO[a], C)
            # B = +mu_elec[v,o]: velocity gauge has positive sign (imaginary perturbation).
            B_elec[:, a] = mu_elec[v, o].reshape(nv * no)
            h_dep_elec[a] = mu_elec

        # Vectorized velocity-gauge electric CPHF solve for all 3 directions simultaneously.
        # ov_sign=+1: imaginary perturbation (U[o,v] = +U[v,o].T, no overlap derivative).
        # dep_sign=-1: B_oo = -h[o,o] + U*A_elec.
        U_E = self._solve_cphf(G_elec, A_elec, B_elec, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=+1, h_dep_all=h_dep_elec, dep_sign=-1)

        APT_HF = np.zeros((natom * 3, 3))
        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                APT_HF[lambda_alpha][beta] += 2 * oe.contract("em,em", U_E[beta][v_, o],
                                                               U_R[lambda_alpha][v_, o]
                                                               + half_S[lambda_alpha][o, v_].T)

        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()
        N = np.zeros((3 * natom, 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * natom):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                N[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        return -2 * APT_HF + N



    def compute_MP2_APTs_VG(self, normalization='full', orbitals='non-canonical', print_level=0):
        """Compute analytic MP2 velocity-gauge atomic polar tensors.

        Parameters
        ----------
        normalization : {'full', 'intermediate'}, optional
        orbitals : {'non-canonical', 'canonical'}, optional
        print_level : int, optional

        Returns
        -------
        ndarray, shape (3*natom, 3)
        """
        wfn_MP2 = mp2_wfn(self.parameters, self.wfn)
        E_MP2, t2 = wfn_MP2.solve_MP2()

        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints

        APT_HF   = np.zeros((natom * 3, 3))
        APT_1    = np.zeros((natom * 3, 3))
        APT_2    = np.zeros((natom * 3, 3))
        APT_3    = np.zeros((natom * 3, 3))
        APT_4    = np.zeros((natom * 3, 3))
        APT_Norm = np.zeros((natom * 3, 3))

        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + oe.contract('ijab,ijab', t2, 2 * t2 - t2.swapaxes(2, 3)))

        # ------------------------------------------------------------------ #
        # Vector-potential (nabla) CPHF: vectorized over 3 directions.       #
        # ------------------------------------------------------------------ #
        A_mom, G_mom = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        nabla_AO = mints.ao_nabla()
        B_mom = np.zeros((nv * no, 3))
        h_dep_mom = np.zeros((3, nbf, nbf))
        for b in range(3):
            nabla_AO[b] = -nabla_AO[b].np
            mu_nabla = oe.contract('mp,mn,nq->pq', np.conjugate(C), nabla_AO[b], C)
            B_mom[:, b] = mu_nabla[v, o].reshape(nv * no)
            h_dep_mom[b] = mu_nabla

        # Imaginary perturbation: ov_sign=+1, dep_sign=-1.
        U_A = self._solve_cphf(G_mom, A_mom, B_mom, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=+1, h_dep_all=h_dep_mom, dep_sign=-1)

        dT2_dA = []
        for b in range(3):
            U_a = U_A[b]
            h_core = h_dep_mom[b]

            df_dA = np.zeros((nbf, nbf))
            df_dA[o, o] -= h_core[o, o].copy()
            df_dA[o, o] += U_a[o, o] * self.wfn.eps[o].reshape(-1, 1) - U_a[o, o].swapaxes(0, 1) * self.wfn.eps[o]
            df_dA[o, o] += oe.contract('em,iejm->ij', U_a[v, o], A_mom.swapaxes(1, 2)[o, v, o, o])
            df_dA[v, v] -= h_core[v, v].copy()
            df_dA[v, v] += U_a[v, v] * self.wfn.eps[v].reshape(-1, 1) - U_a[v, v].swapaxes(0, 1) * self.wfn.eps[v]
            df_dA[v, v] += oe.contract('em,aebm->ab', U_a[v, o], A_mom.swapaxes(1, 2)[v, v, v, o])

            dERI_dA  = oe.contract('tr,pqts->pqrs', U_a[:, t], ERI[t, t, :, t])
            dERI_dA += oe.contract('ts,pqrt->pqrs', U_a[:, t], ERI[t, t, t, :])
            dERI_dA -= oe.contract('tp,tqrs->pqrs', U_a[:, t], ERI[:, t, t, t])
            dERI_dA -= oe.contract('tq,ptrs->pqrs', U_a[:, t], ERI[t, :, t, t])

            dt2_dA = dERI_dA.copy().swapaxes(0, 2).swapaxes(1, 3)[o_, o_, v_, v_]
            dt2_dA += oe.contract('ac,ijcb->ijab', df_dA[v_, v_], t2)
            dt2_dA += oe.contract('bc,ijac->ijab', df_dA[v_, v_], t2)
            dt2_dA -= oe.contract('ki,kjab->ijab', df_dA[o_, o_], t2)
            dt2_dA -= oe.contract('kj,ikab->ijab', df_dA[o_, o_], t2)
            dt2_dA /= wfn_MP2.D_ijab

            dT2_dA.append(dt2_dA)

        # ------------------------------------------------------------------ #
        # Nuclear CPHF: vectorized over all 3*natom displacements.           #
        # ------------------------------------------------------------------ #
        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        npert_nuc = 3 * natom
        B_nuc = np.zeros((nv * no, npert_nuc))
        S_nuc = np.zeros((npert_nuc, nbf, nbf))
        h_dep_nuc = np.zeros((npert_nuc, nbf, nbf))
        h_core_nuc = []
        ERI_core_nuc = []
        half_S = []

        for N1 in atoms:
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                T_core[a] = T_core[a].np
                V_core[a] = V_core[a].np
                S_core[a] = S_core[a].np
                ERI_core[a] = ERI_core[a].np
                ERI_core[a] = ERI_core[a].swapaxes(1, 2)
                half_S_core[a] = half_S_core[a].np

                h_core = T_core[a] + V_core[a]
                F_core = h_core + oe.contract('piqi->pq', 2 * ERI_core[a][:, o, :, o] - ERI_core[a].swapaxes(2, 3)[:, o, :, o])

                k = 3 * N1 + a
                B_nuc[:, k] = (-F_core[v, o]
                                + oe.contract('ai,ii->ai', S_core[a][v, o], F[o, o])
                                + 0.5 * oe.contract('mn,amin->ai', S_core[a][o, o], A.swapaxes(1, 2)[v, o, o, o])
                                ).reshape(nv * no)
                S_nuc[k] = S_core[a]
                h_dep_nuc[k] = F_core
                h_core_nuc.append(h_core)
                ERI_core_nuc.append(ERI_core[a])
                half_S.append(half_S_core[a])

        U_R_list = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                     ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        for N1 in atoms:
            for a in range(3):
                k = 3 * N1 + a
                U_R = U_R_list[k]
                F_core = h_dep_nuc[k]
                S_core_a = S_nuc[k]
                ERI_core_a = ERI_core_nuc[k]
                half_S_core_a = half_S[k]
                lambda_alpha = k

                df_dR = np.zeros((nbf, nbf))
                df_dR[o, o] += F_core[o, o].copy()
                df_dR[o, o] += U_R[o, o] * self.wfn.eps[o].reshape(-1, 1) + U_R[o, o].swapaxes(0, 1) * self.wfn.eps[o]
                df_dR[o, o] += oe.contract('em,iejm->ij', U_R[v, o], A.swapaxes(1, 2)[o, v, o, o])
                df_dR[o, o] -= 0.5 * oe.contract('mn,imjn->ij', S_core_a[o, o], A.swapaxes(1, 2)[o, o, o, o])
                df_dR[v, v] += F_core[v, v].copy()
                df_dR[v, v] += U_R[v, v] * self.wfn.eps[v].reshape(-1, 1) + U_R[v, v].swapaxes(0, 1) * self.wfn.eps[v]
                df_dR[v, v] += oe.contract('em,aebm->ab', U_R[v, o], A.swapaxes(1, 2)[v, v, v, o])
                df_dR[v, v] -= 0.5 * oe.contract('mn,ambn->ab', S_core_a[o, o], A.swapaxes(1, 2)[v, o, v, o])

                dERI_dR = ERI_core_a.copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:, t], ERI[:, t, t, t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:, t], ERI[t, :, t, t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:, t], ERI[t, t, :, t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:, t], ERI[t, t, t, :])

                dt2_dR = dERI_dR.copy()[o_, o_, v_, v_]
                dt2_dR -= oe.contract('kjab,ik->ijab', t2, df_dR[o_, o_])
                dt2_dR -= oe.contract('ikab,kj->ijab', t2, df_dR[o_, o_])
                dt2_dR += oe.contract('ijcb,ac->ijab', t2, df_dR[v_, v_])
                dt2_dR += oe.contract('ijac,cb->ijab', t2, df_dR[v_, v_])
                dt2_dR /= wfn_MP2.D_ijab

                N_R = -(1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2 * t2 - t2.swapaxes(2, 3)))**3))
                N_R *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2 * t2 - t2.swapaxes(2, 3))
                              + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2 * t2 - t2.swapaxes(2, 3))))

                for beta in range(3):
                    if orbitals == 'canonical':
                        APT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        APT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dA[beta])

                        APT_2[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_A[beta][o_, o_])
                        APT_2[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_A[beta][v_, v_])

                        APT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("klcd,mlcd,mk", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        APT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("klcd,kled,ce", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)

                        APT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        APT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ec,ea", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, v_], U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        APT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        APT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        if normalization == 'full':
                            APT_Norm[lambda_alpha][beta] -= N * N_R * 2.0 * oe.contract("ijab,kjab,ki", 2 * t2 - t2.swapaxes(2, 3), t2, U_A[beta][o_, o_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2.0 * oe.contract("ijab,ijcb,ac", 2 * t2 - t2.swapaxes(2, 3), t2, U_A[beta][v_, v_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dA[beta])

                    if orbitals == 'non-canonical':
                        APT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        APT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dA[beta])

                        APT_2[lambda_alpha][beta] += 0

                        APT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        APT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)

                        APT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        APT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        APT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        if normalization == 'full':
                            APT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dA[beta])

        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()
        Nuc = np.zeros((3 * natom, 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * natom):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                Nuc[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        APT_corr = APT_HF + APT_1 + APT_2 + APT_3 + APT_4 + APT_Norm
        return -2 * APT_corr + Nuc



    def compute_CID_APTs_VG(self, normalization='full', orbitals='non-canonical', print_level=0):
        """Compute analytic CID velocity-gauge atomic polar tensors.

        Parameters
        ----------
        normalization : {'full', 'intermediate'}, optional
        orbitals : {'non-canonical', 'canonical'}, optional
        print_level : int, optional

        Returns
        -------
        ndarray, shape (3*natom, 3)
        """
        wfn_CID = ci_wfn(self.parameters, self.wfn)
        E_CID, t2 = wfn_CID.solve_CID()

        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        APT_HF   = np.zeros((natom * 3, 3))
        APT_DD   = np.zeros((natom * 3, 3))
        APT_Norm = np.zeros((natom * 3, 3))

        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + oe.contract('ijab,ijab', t2, 2 * t2 - t2.swapaxes(2, 3)))

        dT2_dA = []
        U_A = []

        N = 1 / np.sqrt(1**2 + oe.contract('ijab,ijab->', np.conjugate(t2), 2 * t2 - t2.swapaxes(2, 3)))
        t0_n = N.copy()
        t2_n = t2 * N

        D_pq = np.zeros_like(F)
        D_pq[o_, o_] -= 2 * oe.contract('jkab,ikab->ij', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), t2_n)
        D_pq[v_, v_] += 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), t2_n)
        D_pq = D_pq[t_, t_]

        D_pqrs = np.zeros_like(ERI)
        D_pqrs[o_, o_, o_, o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2_n), (2 * t2_n - t2_n.swapaxes(2, 3)))
        D_pqrs[v_, v_, v_, v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2_n), (2 * t2_n - t2_n.swapaxes(2, 3)))
        D_pqrs[v_, o_, o_, v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), 2 * t2_n - t2_n.swapaxes(2, 3))
        D_pqrs[v_, o_, v_, o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_, o_, v_, o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_, o_, v_, o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_, o_, v_, o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[o_, o_, v_, v_] += np.conjugate(t0_n) * (2 * t2_n - t2_n.swapaxes(2, 3))
        D_pqrs[v_, v_, o_, o_] += np.conjugate(2 * t2_n.swapaxes(0, 2).swapaxes(1, 3) - t2_n.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3)) * t0_n
        D_pqrs = D_pqrs[t_, t_, t_, t_]

        A_mom, G_mom = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        nabla_AO_raw = mints.ao_nabla()
        nabla_AO = [-nabla_AO_raw[a].np for a in range(3)]

        for a in range(3):
            mu_nabla = oe.contract('mp,mn,nq->pq', np.conjugate(C), nabla_AO[a], C)
            h_core = mu_nabla
            B = h_core[v, o]

            U_a = np.zeros((nbf, nbf))
            U_a[v, o] += (G_mom @ B.reshape((nv * no))).reshape(nv, no)
            U_a[o, v] += U_a[v, o].T

            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1, 1)) + np.eye(no)
                B_oo = -h_core[o, o].copy() + oe.contract('em,iejm->ij', U_a[v, o], A_mom.swapaxes(1, 2)[o, v, o, o])
                U_a[o, o] += B_oo / D
                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1, 1)) + np.eye(nv)
                B_vv = -h_core[v, v].copy() + oe.contract('em,aebm->ab', U_a[v, o], A_mom.swapaxes(1, 2)[v, v, v, o])
                U_a[v, v] += B_vv / D
                for j in range(no):
                    U_a[j, j] = 0
                for c in range(no, nbf):
                    U_a[c, c] = 0

            if orbitals == 'non-canonical':
                U_a[f_, f_] = 0
                U_a[o_, o_] = 0
                U_a[v_, v_] = 0

            df_dA = np.zeros((nbf, nbf))
            df_dA[o, o] -= h_core[o, o].copy()
            df_dA[o, o] += U_a[o, o] * self.wfn.eps[o].reshape(-1, 1) - U_a[o, o].swapaxes(0, 1) * self.wfn.eps[o]
            df_dA[o, o] += oe.contract('em,iejm->ij', U_a[v, o], A_mom.swapaxes(1, 2)[o, v, o, o])
            df_dA[v, v] -= h_core[v, v].copy()
            df_dA[v, v] += U_a[v, v] * self.wfn.eps[v].reshape(-1, 1) - U_a[v, v].swapaxes(0, 1) * self.wfn.eps[v]
            df_dA[v, v] += oe.contract('em,aebm->ab', U_a[v, o], A_mom.swapaxes(1, 2)[v, v, v, o])

            dERI_dA  = oe.contract('tr,pqts->pqrs', U_a[:, t], ERI[t, t, :, t])
            dERI_dA += oe.contract('ts,pqrt->pqrs', U_a[:, t], ERI[t, t, t, :])
            dERI_dA -= oe.contract('tp,tqrs->pqrs', U_a[:, t], ERI[:, t, t, t])
            dERI_dA -= oe.contract('tq,ptrs->pqrs', U_a[:, t], ERI[t, :, t, t])

            dE_dA = oe.contract('pq,pq->', df_dA[t_, t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dA[t_, t_, t_, t_], D_pqrs)
            dE_dA_HF = 2 * oe.contract('ii->', h_core[o, o])

            dt2_dA = -dE_dA * t2
            dt2_dA += oe.contract('ac,ijcb->ijab', df_dA[v_, v_], t2)
            dt2_dA += oe.contract('bc,ijac->ijab', df_dA[v_, v_], t2)
            dt2_dA -= oe.contract('ki,kjab->ijab', df_dA[o_, o_], t2)
            dt2_dA -= oe.contract('kj,ikab->ijab', df_dA[o_, o_], t2)
            dt2_dA += oe.contract('klij,klab->ijab', dERI_dA[o_, o_, o_, o_], t2)
            dt2_dA += oe.contract('abcd,ijcd->ijab', dERI_dA[v_, v_, v_, v_], t2)
            dt2_dA -= oe.contract('kbcj,ikca->ijab', dERI_dA[o_, v_, v_, o_], t2)
            dt2_dA += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
            dt2_dA -= oe.contract('kbic,kjac->ijab', dERI_dA[o_, v_, o_, v_], t2)
            dt2_dA -= oe.contract('kaci,kjbc->ijab', dERI_dA[o_, v_, v_, o_], t2)
            dt2_dA += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
            dt2_dA -= oe.contract('kajc,ikcb->ijab', dERI_dA[o_, v_, o_, v_], t2)
            dt2_dA /= wfn_CID.D_ijab

            dE_dA_proj = oe.contract('ijab,ijab->', t2, 2.0 * dERI_dA[o_, o_, v_, v_] - dERI_dA.swapaxes(2, 3)[o_, o_, v_, v_])
            dE_dA_proj += oe.contract('ijab,ijab->', dt2_dA, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])
            dt2_dA = dt2_dA.copy()

            iteration = 1
            while iteration <= self.parameters['max_iterations']:
                dE_dA_proj_old = dE_dA_proj
                dt2_dA_old = dt2_dA.copy()

                dRt2_dA = dERI_dA.copy().swapaxes(0, 2).swapaxes(1, 3)[o_, o_, v_, v_]
                dRt2_dA -= dE_dA_proj * t2
                dRt2_dA += oe.contract('ac,ijcb->ijab', df_dA[v_, v_], t2)
                dRt2_dA += oe.contract('bc,ijac->ijab', df_dA[v_, v_], t2)
                dRt2_dA -= oe.contract('ki,kjab->ijab', df_dA[o_, o_], t2)
                dRt2_dA -= oe.contract('kj,ikab->ijab', df_dA[o_, o_], t2)
                dRt2_dA += oe.contract('klij,klab->ijab', dERI_dA[o_, o_, o_, o_], t2)
                dRt2_dA += oe.contract('abcd,ijcd->ijab', dERI_dA[v_, v_, v_, v_], t2)
                dRt2_dA -= oe.contract('kbcj,ikca->ijab', dERI_dA[o_, v_, v_, o_], t2)
                dRt2_dA += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dRt2_dA -= oe.contract('kbic,kjac->ijab', dERI_dA[o_, v_, o_, v_], t2)
                dRt2_dA -= oe.contract('kaci,kjbc->ijab', dERI_dA[o_, v_, v_, o_], t2)
                dRt2_dA += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dRt2_dA -= oe.contract('kajc,ikcb->ijab', dERI_dA[o_, v_, o_, v_], t2)
                dRt2_dA -= E_CID * dt2_dA
                dRt2_dA += oe.contract('ac,ijcb->ijab', F[v_, v_], dt2_dA)
                dRt2_dA += oe.contract('bc,ijac->ijab', F[v_, v_], dt2_dA)
                dRt2_dA -= oe.contract('ki,kjab->ijab', F[o_, o_], dt2_dA)
                dRt2_dA -= oe.contract('kj,ikab->ijab', F[o_, o_], dt2_dA)
                dRt2_dA += oe.contract('klij,klab->ijab', ERI[o_, o_, o_, o_], dt2_dA)
                dRt2_dA += oe.contract('abcd,ijcd->ijab', ERI[v_, v_, v_, v_], dt2_dA)
                dRt2_dA -= oe.contract('kbcj,ikca->ijab', ERI[o_, v_, v_, o_], dt2_dA)
                dRt2_dA += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dA)
                dRt2_dA -= oe.contract('kbic,kjac->ijab', ERI[o_, v_, o_, v_], dt2_dA)
                dRt2_dA -= oe.contract('kaci,kjbc->ijab', ERI[o_, v_, v_, o_], dt2_dA)
                dRt2_dA += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dA)
                dRt2_dA -= oe.contract('kajc,ikcb->ijab', ERI[o_, v_, o_, v_], dt2_dA)

                dt2_dA += dRt2_dA / wfn_CID.D_ijab

                if self.parameters['DIIS']:
                    occ = len(dt2_dA)
                    vir = len(dt2_dA[0][0])
                    dt2_dA_flat = len(np.reshape(dt2_dA, (-1)))
                    res_vec = np.reshape(dRt2_dA, (-1))
                    t_vec = np.reshape(dt2_dA, (-1))
                    if iteration == 1:
                        t_iter = np.atleast_2d(t_vec).T
                        e_iter = np.atleast_2d(res_vec).T
                    t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                    dt2_dA = np.reshape(t_vec, (occ, occ, vir, vir))

                dE_dA_proj = oe.contract('ijab,ijab->', t2, 2.0 * dERI_dA[o_, o_, v_, v_] - dERI_dA.swapaxes(2, 3)[o_, o_, v_, v_])
                dE_dA_proj += oe.contract('ijab,ijab->', dt2_dA, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])

                rms_dt2_dA = np.sqrt(oe.contract('ijab,ijab->', dt2_dA_old - dt2_dA, dt2_dA_old - dt2_dA))
                delta_dE_dA_proj = dE_dA_proj_old - dE_dA_proj

                if iteration > 1:
                    if abs(delta_dE_dA_proj) < self.parameters['e_convergence'] and rms_dt2_dA < self.parameters['d_convergence']:
                        break
                if iteration == self.parameters['max_iterations']:
                    if abs(delta_dE_dA_proj) > self.parameters['e_convergence'] or rms_dt2_dA > self.parameters['d_convergence']:
                        print("Not converged.")
                iteration += 1

            dT2_dA.append(dt2_dA)
            U_A.append(U_a)

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        for N1 in atoms:
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                T_core[a] = T_core[a].np
                V_core[a] = V_core[a].np
                S_core[a] = S_core[a].np
                ERI_core[a] = ERI_core[a].np
                ERI_core[a] = ERI_core[a].swapaxes(1, 2)
                half_S_core[a] = half_S_core[a].np

                h_core = T_core[a] + V_core[a]
                F_core = h_core + oe.contract('piqi->pq', 2 * ERI_core[a][:, o, :, o] - ERI_core[a].swapaxes(2, 3)[:, o, :, o])
                B = -F_core[v, o] + oe.contract('ai,ii->ai', S_core[a][v, o], F[o, o]) + 0.5 * oe.contract('mn,amin->ai', S_core[a][o, o], A.swapaxes(1, 2)[v, o, o, o])

                U_R = np.zeros((nbf, nbf))
                U_R[v, o] += (G @ B.reshape((nv * no))).reshape(nv, no)
                U_R[o, v] -= U_R[v, o].T + S_core[a][o, v]

                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1, 1)) + np.eye(no)
                    B_oo = F_core[o, o].copy() - oe.contract('ij,jj->ij', S_core[a][o, o], F[o, o]) + oe.contract('em,iejm->ij', U_R[v, o], A.swapaxes(1, 2)[o, v, o, o]) - 0.5 * oe.contract('mn,imjn->ij', S_core[a][o, o], A.swapaxes(1, 2)[o, o, o, o])
                    U_R[o, o] += B_oo / D
                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1, 1)) + np.eye(nv)
                    B_vv = F_core[v, v].copy() - oe.contract('ab,bb->ab', S_core[a][v, v], F[v, v]) + oe.contract('em,aebm->ab', U_R[v, o], A.swapaxes(1, 2)[v, v, v, o]) - 0.5 * oe.contract('mn,ambn->ab', S_core[a][o, o], A.swapaxes(1, 2)[v, o, v, o])
                    U_R[v, v] += B_vv / D
                    for j in range(no):
                        U_R[j, j] = -0.5 * S_core[a][j, j]
                    for c in range(no, nbf):
                        U_R[c, c] = -0.5 * S_core[a][c, c]

                if orbitals == 'non-canonical':
                    U_R[f_, f_] = -0.5 * S_core[a][f_, f_]
                    U_R[o_, o_] = -0.5 * S_core[a][o_, o_]
                    U_R[v_, v_] = -0.5 * S_core[a][v_, v_]

                df_dR = np.zeros((nbf, nbf))
                df_dR[o, o] += F_core[o, o].copy()
                df_dR[o, o] += U_R[o, o] * self.wfn.eps[o].reshape(-1, 1) + U_R[o, o].swapaxes(0, 1) * self.wfn.eps[o]
                df_dR[o, o] += oe.contract('em,iejm->ij', U_R[v, o], A.swapaxes(1, 2)[o, v, o, o])
                df_dR[o, o] -= 0.5 * oe.contract('mn,imjn->ij', S_core[a][o, o], A.swapaxes(1, 2)[o, o, o, o])
                df_dR[v, v] += F_core[v, v].copy()
                df_dR[v, v] += U_R[v, v] * self.wfn.eps[v].reshape(-1, 1) + U_R[v, v].swapaxes(0, 1) * self.wfn.eps[v]
                df_dR[v, v] += oe.contract('em,aebm->ab', U_R[v, o], A.swapaxes(1, 2)[v, v, v, o])
                df_dR[v, v] -= 0.5 * oe.contract('mn,ambn->ab', S_core[a][o, o], A.swapaxes(1, 2)[v, o, v, o])

                dERI_dR = ERI_core[a].copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:, t], ERI[:, t, t, t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:, t], ERI[t, :, t, t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:, t], ERI[t, t, :, t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:, t], ERI[t, t, t, :])

                dE_dR = oe.contract('pq,pq->', df_dR[t_, t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dR[t_, t_, t_, t_], D_pqrs)
                dE_dR_HF = 2 * oe.contract('ii->', h_core[o, o])
                dE_dR_HF += oe.contract('ijij->', 2 * ERI_core[a][o, o, o, o] - ERI_core[a].swapaxes(2, 3)[o, o, o, o])
                dE_dR_HF -= 2 * oe.contract('ii,i->', S_core[a][o, o], self.wfn.eps[o])
                dE_dR_HF += Nuc_Gradient[N1][a]

                dt2_dR = -dE_dR * t2
                dt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_, v_], t2)
                dt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_, v_], t2)
                dt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_, o_], t2)
                dt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_, o_], t2)
                dt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_, o_, o_, o_], t2)
                dt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_, v_, v_, v_], t2)
                dt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_, v_, v_, o_], t2)
                dt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_, v_, o_, v_], t2)
                dt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_, v_, v_, o_], t2)
                dt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_, v_, o_, v_], t2)
                dt2_dR /= wfn_CID.D_ijab

                dE_dR_proj = oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_, o_, v_, v_] - dERI_dR.swapaxes(2, 3)[o_, o_, v_, v_])
                dE_dR_proj += oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])
                dt2_dR = dt2_dR.copy()

                iteration = 1
                while iteration <= self.parameters['max_iterations']:
                    dE_dR_proj_old = dE_dR_proj
                    dt2_dR_old = dt2_dR.copy()

                    dRt2_dR = dERI_dR.copy().swapaxes(0, 2).swapaxes(1, 3)[o_, o_, v_, v_]
                    dRt2_dR -= dE_dR_proj * t2
                    dRt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_, v_], t2)
                    dRt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_, v_], t2)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_, o_], t2)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_, o_], t2)
                    dRt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_, o_, o_, o_], t2)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_, v_, v_, v_], t2)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_, v_, v_, o_], t2)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_, v_, o_, v_], t2)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_, v_, v_, o_], t2)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_, v_, o_, v_], t2)
                    dRt2_dR -= E_CID * dt2_dR
                    dRt2_dR += oe.contract('ac,ijcb->ijab', F[v_, v_], dt2_dR)
                    dRt2_dR += oe.contract('bc,ijac->ijab', F[v_, v_], dt2_dR)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', F[o_, o_], dt2_dR)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', F[o_, o_], dt2_dR)
                    dRt2_dR += oe.contract('klij,klab->ijab', ERI[o_, o_, o_, o_], dt2_dR)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', ERI[v_, v_, v_, v_], dt2_dR)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', ERI[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', ERI[o_, v_, o_, v_], dt2_dR)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', ERI[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', ERI[o_, v_, o_, v_], dt2_dR)

                    dt2_dR += dRt2_dR / wfn_CID.D_ijab

                    if self.parameters['DIIS']:
                        occ = len(dt2_dR)
                        vir = len(dt2_dR[0][0])
                        dt2_dR_flat = len(np.reshape(dt2_dR, (-1)))
                        res_vec = np.reshape(dRt2_dR, (-1))
                        t_vec = np.reshape(dt2_dR, (-1))
                        if iteration == 1:
                            t_iter = np.atleast_2d(t_vec).T
                            e_iter = np.atleast_2d(res_vec).T
                        t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                        dt2_dR = np.reshape(t_vec, (occ, occ, vir, vir))

                    dE_dR_proj = oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_, o_, v_, v_] - dERI_dR.swapaxes(2, 3)[o_, o_, v_, v_])
                    dE_dR_proj += oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])

                    rms_dt2_dR = np.sqrt(oe.contract('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR))
                    delta_dE_dR_proj = dE_dR_proj_old - dE_dR_proj

                    if iteration > 1:
                        if abs(delta_dE_dR_proj) < self.parameters['e_convergence'] and rms_dt2_dR < self.parameters['d_convergence']:
                            break
                    if iteration == self.parameters['max_iterations']:
                        if abs(delta_dE_dR_proj) > self.parameters['e_convergence'] or rms_dt2_dR > self.parameters['d_convergence']:
                            print("Not converged.")
                    iteration += 1

                N_R = -(1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2 * t2 - t2.swapaxes(2, 3)))**3))
                N_R *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2 * t2 - t2.swapaxes(2, 3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2 * t2 - t2.swapaxes(2, 3))))

                for beta in range(3):
                    lambda_alpha = 3 * N1 + a

                    if orbitals == 'canonical':
                        APT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_A[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        APT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dA[beta])
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_A[beta][o_, o_])
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_A[beta][v_, v_])
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("klcd,mlcd,mk", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,kled,ce", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ec,ea", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, v_], U_R[v_, v_] + half_S_core[a][v_, v_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        if normalization == 'full':
                            APT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("ijab,kjab,ki", 2 * t2 - t2.swapaxes(2, 3), t2, U_A[beta][o_, o_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,ijcb,ac", 2 * t2 - t2.swapaxes(2, 3), t2, U_A[beta][v_, v_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dA[beta])

                    if orbitals == 'non-canonical':
                        APT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_A[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        APT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dA[beta])
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        if normalization == 'full':
                            APT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dA[beta])

        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()
        Nuc = np.zeros((3 * natom, 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * natom):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                Nuc[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        APT_total = APT_HF + APT_DD + APT_Norm
        return -2 * APT_total + Nuc



    def compute_CISD_APTs_VG(self, normalization='full', orbitals='non-canonical', print_level=0):
        """Compute analytic CISD velocity-gauge atomic polar tensors.

        Parameters
        ----------
        normalization : {'full', 'intermediate'}, optional
        orbitals : {'non-canonical', 'canonical'}, optional
        print_level : int, optional

        Returns
        -------
        ndarray, shape (3*natom, 3)
        """
        wfn_CISD = ci_wfn(self.parameters, self.wfn)
        E_CISD, t1, t2 = wfn_CISD.solve_CISD()

        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        APT_HF   = np.zeros((natom * 3, 3))
        APT_S0   = np.zeros((natom * 3, 3))
        APT_0S   = np.zeros((natom * 3, 3))
        APT_SS   = np.zeros((natom * 3, 3))
        APT_DS   = np.zeros((natom * 3, 3))
        APT_SD   = np.zeros((natom * 3, 3))
        APT_DD   = np.zeros((natom * 3, 3))
        APT_Norm = np.zeros((natom * 3, 3))

        N = 1 / np.sqrt(1**2 + 2 * oe.contract('ia,ia->', np.conjugate(t1), t1) + oe.contract('ijab,ijab->', np.conjugate(t2), 2 * t2 - t2.swapaxes(2, 3)))
        t0_n = N.copy()
        t1_n = t1 * N
        t2_n = t2 * N

        D_pq = np.zeros_like(F)
        D_pq[o_, o_] -= 2 * oe.contract('ja,ia->ij', np.conjugate(t1_n), t1_n) + 2 * oe.contract('jkab,ikab->ij', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), t2_n)
        D_pq[v_, v_] += 2 * oe.contract('ia,ib->ab', np.conjugate(t1_n), t1_n) + 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), t2_n)
        D_pq[o_, v_] += 2 * np.conjugate(t0_n) * t1_n + 2 * oe.contract('jb,ijab->ia', np.conjugate(t1_n), t2_n - t2_n.swapaxes(2, 3))
        D_pq[v_, o_] += 2 * np.conjugate(t1_n.T) * t0_n + 2 * oe.contract('ijab,jb->ai', np.conjugate(t2_n - t2_n.swapaxes(2, 3)), t1_n)
        D_pq = D_pq[t_, t_]

        D_pqrs = np.zeros_like(ERI)
        D_pqrs[o_, o_, o_, o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2_n), (2 * t2_n - t2_n.swapaxes(2, 3)))
        D_pqrs[v_, v_, v_, v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2_n), (2 * t2_n - t2_n.swapaxes(2, 3)))
        D_pqrs[o_, v_, v_, o_] += 4 * oe.contract('ja,ib->iabj', np.conjugate(t1_n), t1_n)
        D_pqrs[o_, v_, o_, v_] -= 2 * oe.contract('ja,ib->iajb', np.conjugate(t1_n), t1_n)
        D_pqrs[v_, o_, o_, v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), 2 * t2_n - t2_n.swapaxes(2, 3))
        D_pqrs[v_, o_, v_, o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_, o_, v_, o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_, o_, v_, o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_, o_, v_, o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[o_, o_, v_, v_] += np.conjugate(t0_n) * (2 * t2_n - t2_n.swapaxes(2, 3))
        D_pqrs[v_, v_, o_, o_] += np.conjugate(2 * t2_n.swapaxes(0, 2).swapaxes(1, 3) - t2_n.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3)) * t0_n
        D_pqrs[v_, o_, v_, v_] += 2 * oe.contract('ja,ijcb->aibc', np.conjugate(t1_n), 2 * t2_n - t2_n.swapaxes(2, 3))
        D_pqrs[o_, v_, o_, o_] -= 2 * oe.contract('kjab,ib->iajk', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), t1_n)
        D_pqrs[v_, v_, v_, o_] += 2 * oe.contract('jiab,jc->abci', np.conjugate(2 * t2_n - t2_n.swapaxes(2, 3)), t1_n)
        D_pqrs[o_, o_, o_, v_] -= 2 * oe.contract('kb,ijba->ijka', np.conjugate(t1_n), 2 * t2_n - t2_n.swapaxes(2, 3))
        D_pqrs = D_pqrs[t_, t_, t_, t_]

        A_mom, G_mom = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        nabla_AO_raw = mints.ao_nabla()
        B_mom = np.zeros((nv * no, 3))
        h_dep_mom = np.zeros((3, nbf, nbf))
        for a in range(3):
            nabla_AO_a = -nabla_AO_raw[a].np
            mu_nabla = oe.contract('mp,mn,nq->pq', np.conjugate(C), nabla_AO_a, C)
            B_mom[:, a] = mu_nabla[v, o].reshape(nv * no)
            h_dep_mom[a] = mu_nabla

        U_A = self._solve_cphf(G_mom, A_mom, B_mom, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=+1, h_dep_all=h_dep_mom, dep_sign=-1)

        dT1_dA = []
        dT2_dA = []

        for a in range(3):
            h_core = h_dep_mom[a]
            U_a = U_A[a]

            df_dA = np.zeros((nbf, nbf))
            df_dA[o, o] -= h_core[o, o].copy()
            df_dA[o, o] += U_a[o, o] * self.wfn.eps[o].reshape(-1, 1) - U_a[o, o].swapaxes(0, 1) * self.wfn.eps[o]
            df_dA[o, o] += oe.contract('em,iejm->ij', U_a[v, o], A_mom.swapaxes(1, 2)[o, v, o, o])
            df_dA[v, v] -= h_core[v, v].copy()
            df_dA[v, v] += U_a[v, v] * self.wfn.eps[v].reshape(-1, 1) - U_a[v, v].swapaxes(0, 1) * self.wfn.eps[v]
            df_dA[v, v] += oe.contract('em,aebm->ab', U_a[v, o], A_mom.swapaxes(1, 2)[v, v, v, o])

            dERI_dA  = oe.contract('tr,pqts->pqrs', U_a[:, t], ERI[t, t, :, t])
            dERI_dA += oe.contract('ts,pqrt->pqrs', U_a[:, t], ERI[t, t, t, :])
            dERI_dA -= oe.contract('tp,tqrs->pqrs', U_a[:, t], ERI[:, t, t, t])
            dERI_dA -= oe.contract('tq,ptrs->pqrs', U_a[:, t], ERI[t, :, t, t])

            dE_dA = oe.contract('pq,pq->', df_dA[t_, t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dA[t_, t_, t_, t_], D_pqrs)
            dE_dA_HF = 2 * oe.contract('ii->', h_core[o, o])

            dt1_dA = -dE_dA * t1
            dt1_dA -= oe.contract('ji,ja->ia', df_dA[o_, o_], t1)
            dt1_dA += oe.contract('ab,ib->ia', df_dA[v_, v_], t1)
            dt1_dA += oe.contract('jabi,jb->ia', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t1)
            dt1_dA += oe.contract('jb,ijab->ia', df_dA[o_, v_], 2.0 * t2 - t2.swapaxes(2, 3))
            dt1_dA += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dA[v_, o_, v_, v_] - dERI_dA.swapaxes(2, 3)[v_, o_, v_, v_], t2)
            dt1_dA -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dA[o_, o_, o_, v_] - dERI_dA.swapaxes(2, 3)[o_, o_, o_, v_], t2)
            dt1_dA /= wfn_CISD.D_ia

            dt2_dA = -dE_dA * t2
            dt2_dA += oe.contract('abcj,ic->ijab', dERI_dA[v_, v_, v_, o_], t1)
            dt2_dA += oe.contract('abic,jc->ijab', dERI_dA[v_, v_, o_, v_], t1)
            dt2_dA -= oe.contract('kbij,ka->ijab', dERI_dA[o_, v_, o_, o_], t1)
            dt2_dA -= oe.contract('akij,kb->ijab', dERI_dA[v_, o_, o_, o_], t1)
            dt2_dA += oe.contract('ac,ijcb->ijab', df_dA[v_, v_], t2)
            dt2_dA += oe.contract('bc,ijac->ijab', df_dA[v_, v_], t2)
            dt2_dA -= oe.contract('ki,kjab->ijab', df_dA[o_, o_], t2)
            dt2_dA -= oe.contract('kj,ikab->ijab', df_dA[o_, o_], t2)
            dt2_dA += oe.contract('klij,klab->ijab', dERI_dA[o_, o_, o_, o_], t2)
            dt2_dA += oe.contract('abcd,ijcd->ijab', dERI_dA[v_, v_, v_, v_], t2)
            dt2_dA -= oe.contract('kbcj,ikca->ijab', dERI_dA[o_, v_, v_, o_], t2)
            dt2_dA += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
            dt2_dA -= oe.contract('kbic,kjac->ijab', dERI_dA[o_, v_, o_, v_], t2)
            dt2_dA -= oe.contract('kaci,kjbc->ijab', dERI_dA[o_, v_, v_, o_], t2)
            dt2_dA += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
            dt2_dA -= oe.contract('kajc,ikcb->ijab', dERI_dA[o_, v_, o_, v_], t2)
            dt2_dA /= wfn_CISD.D_ijab

            dE_dA_proj  = 2.0 * oe.contract('ia,ia->', t1, df_dA[o_, v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dA[o_, o_, v_, v_] - dERI_dA.swapaxes(2, 3)[o_, o_, v_, v_])
            dE_dA_proj += 2.0 * oe.contract('ia,ia->', dt1_dA, F[o_, v_]) + oe.contract('ijab,ijab->', dt2_dA, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])
            dt1_dA = dt1_dA.copy()
            dt2_dA = dt2_dA.copy()

            iteration = 1
            while iteration <= self.parameters['max_iterations']:
                dE_dA_proj_old = dE_dA_proj
                dt1_dA_old = dt1_dA.copy()
                dt2_dA_old = dt2_dA.copy()

                dRt1_dA = df_dA.copy().swapaxes(0, 1)[o_, v_]
                dRt1_dA -= dE_dA_proj * t1
                dRt1_dA -= oe.contract('ji,ja->ia', df_dA[o_, o_], t1)
                dRt1_dA += oe.contract('ab,ib->ia', df_dA[v_, v_], t1)
                dRt1_dA += oe.contract('jabi,jb->ia', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t1)
                dRt1_dA += oe.contract('jb,ijab->ia', df_dA[o_, v_], 2.0 * t2 - t2.swapaxes(2, 3))
                dRt1_dA += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dA[v_, o_, v_, v_] - dERI_dA.swapaxes(2, 3)[v_, o_, v_, v_], t2)
                dRt1_dA -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dA[o_, o_, o_, v_] - dERI_dA.swapaxes(2, 3)[o_, o_, o_, v_], t2)
                dRt1_dA -= E_CISD * dt1_dA
                dRt1_dA -= oe.contract('ji,ja->ia', F[o_, o_], dt1_dA)
                dRt1_dA += oe.contract('ab,ib->ia', F[v_, v_], dt1_dA)
                dRt1_dA += oe.contract('jabi,jb->ia', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt1_dA)
                dRt1_dA += oe.contract('jb,ijab->ia', F[o_, v_], 2.0 * dt2_dA - dt2_dA.swapaxes(2, 3))
                dRt1_dA += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_, o_, v_, v_] - ERI.swapaxes(2, 3)[v_, o_, v_, v_], dt2_dA)
                dRt1_dA -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_, o_, o_, v_] - ERI.swapaxes(2, 3)[o_, o_, o_, v_], dt2_dA)

                dRt2_dA = dERI_dA.copy().swapaxes(0, 2).swapaxes(1, 3)[o_, o_, v_, v_]
                dRt2_dA -= dE_dA_proj * t2
                dRt2_dA += oe.contract('abcj,ic->ijab', dERI_dA[v_, v_, v_, o_], t1)
                dRt2_dA += oe.contract('abic,jc->ijab', dERI_dA[v_, v_, o_, v_], t1)
                dRt2_dA -= oe.contract('kbij,ka->ijab', dERI_dA[o_, v_, o_, o_], t1)
                dRt2_dA -= oe.contract('akij,kb->ijab', dERI_dA[v_, o_, o_, o_], t1)
                dRt2_dA += oe.contract('ac,ijcb->ijab', df_dA[v_, v_], t2)
                dRt2_dA += oe.contract('bc,ijac->ijab', df_dA[v_, v_], t2)
                dRt2_dA -= oe.contract('ki,kjab->ijab', df_dA[o_, o_], t2)
                dRt2_dA -= oe.contract('kj,ikab->ijab', df_dA[o_, o_], t2)
                dRt2_dA += oe.contract('klij,klab->ijab', dERI_dA[o_, o_, o_, o_], t2)
                dRt2_dA += oe.contract('abcd,ijcd->ijab', dERI_dA[v_, v_, v_, v_], t2)
                dRt2_dA -= oe.contract('kbcj,ikca->ijab', dERI_dA[o_, v_, v_, o_], t2)
                dRt2_dA += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dRt2_dA -= oe.contract('kbic,kjac->ijab', dERI_dA[o_, v_, o_, v_], t2)
                dRt2_dA -= oe.contract('kaci,kjbc->ijab', dERI_dA[o_, v_, v_, o_], t2)
                dRt2_dA += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dA[o_, v_, v_, o_] - dERI_dA.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dRt2_dA -= oe.contract('kajc,ikcb->ijab', dERI_dA[o_, v_, o_, v_], t2)
                dRt2_dA -= E_CISD * dt2_dA
                dRt2_dA += oe.contract('abcj,ic->ijab', ERI[v_, v_, v_, o_], dt1_dA)
                dRt2_dA += oe.contract('abic,jc->ijab', ERI[v_, v_, o_, v_], dt1_dA)
                dRt2_dA -= oe.contract('kbij,ka->ijab', ERI[o_, v_, o_, o_], dt1_dA)
                dRt2_dA -= oe.contract('akij,kb->ijab', ERI[v_, o_, o_, o_], dt1_dA)
                dRt2_dA += oe.contract('ac,ijcb->ijab', F[v_, v_], dt2_dA)
                dRt2_dA += oe.contract('bc,ijac->ijab', F[v_, v_], dt2_dA)
                dRt2_dA -= oe.contract('ki,kjab->ijab', F[o_, o_], dt2_dA)
                dRt2_dA -= oe.contract('kj,ikab->ijab', F[o_, o_], dt2_dA)
                dRt2_dA += oe.contract('klij,klab->ijab', ERI[o_, o_, o_, o_], dt2_dA)
                dRt2_dA += oe.contract('abcd,ijcd->ijab', ERI[v_, v_, v_, v_], dt2_dA)
                dRt2_dA -= oe.contract('kbcj,ikca->ijab', ERI[o_, v_, v_, o_], dt2_dA)
                dRt2_dA += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dA)
                dRt2_dA -= oe.contract('kbic,kjac->ijab', ERI[o_, v_, o_, v_], dt2_dA)
                dRt2_dA -= oe.contract('kaci,kjbc->ijab', ERI[o_, v_, v_, o_], dt2_dA)
                dRt2_dA += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dA)
                dRt2_dA -= oe.contract('kajc,ikcb->ijab', ERI[o_, v_, o_, v_], dt2_dA)

                dt1_dA += dRt1_dA / wfn_CISD.D_ia
                dt2_dA += dRt2_dA / wfn_CISD.D_ijab

                if self.parameters['DIIS']:
                    occ = len(dt1_dA)
                    vir = len(dt1_dA[0])
                    dt1_dA_flat = len(np.reshape(dt1_dA, (-1)))
                    dt2_dA_flat = len(np.reshape(dt2_dA, (-1)))
                    res_vec = np.concatenate((np.reshape(dRt1_dA, (-1)), np.reshape(dRt2_dA, (-1))))
                    t_vec = np.concatenate((np.reshape(dt1_dA, (-1)), np.reshape(dt2_dA, (-1))))
                    if iteration == 1:
                        t_iter = np.atleast_2d(t_vec).T
                        e_iter = np.atleast_2d(res_vec).T
                    t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                    dt1_dA = np.reshape(t_vec[0:dt1_dA_flat], (occ, vir))
                    dt2_dA = np.reshape(t_vec[dt1_dA_flat:], (occ, occ, vir, vir))

                dE_dA_proj  = 2.0 * oe.contract('ia,ia->', t1, df_dA[o_, v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dA[o_, o_, v_, v_] - dERI_dA.swapaxes(2, 3)[o_, o_, v_, v_])
                dE_dA_proj += 2.0 * oe.contract('ia,ia->', dt1_dA, F[o_, v_]) + oe.contract('ijab,ijab->', dt2_dA, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])

                rms_dt1_dA = np.sqrt(oe.contract('ia,ia->', dt1_dA_old - dt1_dA, dt1_dA_old - dt1_dA))
                rms_dt2_dA = np.sqrt(oe.contract('ijab,ijab->', dt2_dA_old - dt2_dA, dt2_dA_old - dt2_dA))
                delta_dE_dA_proj = dE_dA_proj_old - dE_dA_proj

                if iteration > 1:
                    if abs(delta_dE_dA_proj) < self.parameters['e_convergence'] and rms_dt1_dA < self.parameters['d_convergence'] and rms_dt2_dA < self.parameters['d_convergence']:
                        break
                if iteration == self.parameters['max_iterations']:
                    if abs(delta_dE_dA_proj) > self.parameters['e_convergence'] or rms_dt1_dA > self.parameters['d_convergence'] or rms_dt2_dA > self.parameters['d_convergence']:
                        print("Not converged.")
                iteration += 1

            dT1_dA.append(dt1_dA)
            dT2_dA.append(dt2_dA)

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        npert_nuc = 3 * natom
        B_nuc = np.zeros((nv * no, npert_nuc))
        S_nuc = np.zeros((npert_nuc, nbf, nbf))
        h_dep_nuc = np.zeros((npert_nuc, nbf, nbf))
        h_core_store = []
        ERI_core_store = []
        half_S_core_store = []

        for N1 in atoms:
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_arr = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_arr = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            half_S_arr = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                k = 3 * N1 + a
                T_core[a] = T_core[a].np
                V_core[a] = V_core[a].np
                S_arr[a] = S_arr[a].np
                ERI_arr[a] = ERI_arr[a].np
                ERI_arr[a] = ERI_arr[a].swapaxes(1, 2)
                half_S_arr[a] = half_S_arr[a].np

                h_core = T_core[a] + V_core[a]
                F_core = h_core + oe.contract('piqi->pq', 2 * ERI_arr[a][:, o, :, o] - ERI_arr[a].swapaxes(2, 3)[:, o, :, o])

                B_nuc[:, k] = (-F_core[v, o]
                               + oe.contract('ai,ii->ai', S_arr[a][v, o], F[o, o])
                               + 0.5 * oe.contract('mn,amin->ai', S_arr[a][o, o], A.swapaxes(1, 2)[v, o, o, o])
                               ).reshape(nv * no)
                S_nuc[k] = S_arr[a]
                h_dep_nuc[k] = F_core
                h_core_store.append(h_core)
                ERI_core_store.append(ERI_arr[a].copy())
                half_S_core_store.append(half_S_arr[a])

        U_R_list = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                     ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        for N1 in atoms:
            for a in range(3):
                k = 3 * N1 + a
                U_R = U_R_list[k]
                h_core = h_core_store[k]
                F_core = h_dep_nuc[k]
                S_core_a = S_nuc[k]
                ERI_core_a = ERI_core_store[k]
                half_S_core_a = half_S_core_store[k]

                df_dR = np.zeros((nbf, nbf))
                df_dR[o, o] += F_core[o, o].copy()
                df_dR[o, o] += U_R[o, o] * self.wfn.eps[o].reshape(-1, 1) + U_R[o, o].swapaxes(0, 1) * self.wfn.eps[o]
                df_dR[o, o] += oe.contract('em,iejm->ij', U_R[v, o], A.swapaxes(1, 2)[o, v, o, o])
                df_dR[o, o] -= 0.5 * oe.contract('mn,imjn->ij', S_core_a[o, o], A.swapaxes(1, 2)[o, o, o, o])
                df_dR[v, v] += F_core[v, v].copy()
                df_dR[v, v] += U_R[v, v] * self.wfn.eps[v].reshape(-1, 1) + U_R[v, v].swapaxes(0, 1) * self.wfn.eps[v]
                df_dR[v, v] += oe.contract('em,aebm->ab', U_R[v, o], A.swapaxes(1, 2)[v, v, v, o])
                df_dR[v, v] -= 0.5 * oe.contract('mn,ambn->ab', S_core_a[o, o], A.swapaxes(1, 2)[v, o, v, o])

                dERI_dR = ERI_core_a.copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:, t], ERI[:, t, t, t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:, t], ERI[t, :, t, t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:, t], ERI[t, t, :, t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:, t], ERI[t, t, t, :])

                dE_dR = oe.contract('pq,pq->', df_dR[t_, t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dR[t_, t_, t_, t_], D_pqrs)
                dE_dR_HF = 2 * oe.contract('ii->', h_core[o, o])
                dE_dR_HF += oe.contract('ijij->', 2 * ERI_core_a[o, o, o, o] - ERI_core_a.swapaxes(2, 3)[o, o, o, o])
                dE_dR_HF -= 2 * oe.contract('ii,i->', S_core_a[o, o], self.wfn.eps[o])
                dE_dR_HF += Nuc_Gradient[N1][a]

                dt1_dR = -dE_dR * t1
                dt1_dR -= oe.contract('ji,ja->ia', df_dR[o_, o_], t1)
                dt1_dR += oe.contract('ab,ib->ia', df_dR[v_, v_], t1)
                dt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t1)
                dt1_dR += oe.contract('jb,ijab->ia', df_dR[o_, v_], 2.0 * t2 - t2.swapaxes(2, 3))
                dt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_, o_, v_, v_] - dERI_dR.swapaxes(2, 3)[v_, o_, v_, v_], t2)
                dt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_, o_, o_, v_] - dERI_dR.swapaxes(2, 3)[o_, o_, o_, v_], t2)
                dt1_dR /= wfn_CISD.D_ia

                dt2_dR = -dE_dR * t2
                dt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_, v_, v_, o_], t1)
                dt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_, v_, o_, v_], t1)
                dt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_, v_, o_, o_], t1)
                dt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_, o_, o_, o_], t1)
                dt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_, v_], t2)
                dt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_, v_], t2)
                dt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_, o_], t2)
                dt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_, o_], t2)
                dt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_, o_, o_, o_], t2)
                dt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_, v_, v_, v_], t2)
                dt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_, v_, v_, o_], t2)
                dt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_, v_, o_, v_], t2)
                dt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_, v_, v_, o_], t2)
                dt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                dt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_, v_, o_, v_], t2)
                dt2_dR /= wfn_CISD.D_ijab

                dE_dR_proj  = 2.0 * oe.contract('ia,ia->', t1, df_dR[o_, v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_, o_, v_, v_] - dERI_dR.swapaxes(2, 3)[o_, o_, v_, v_])
                dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_, v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])
                dt1_dR = dt1_dR.copy()
                dt2_dR = dt2_dR.copy()

                iteration = 1
                while iteration <= self.parameters['max_iterations']:
                    dE_dR_proj_old = dE_dR_proj
                    dt1_dR_old = dt1_dR.copy()
                    dt2_dR_old = dt2_dR.copy()

                    dRt1_dR = df_dR.copy().swapaxes(0, 1)[o_, v_]
                    dRt1_dR -= dE_dR_proj * t1
                    dRt1_dR -= oe.contract('ji,ja->ia', df_dR[o_, o_], t1)
                    dRt1_dR += oe.contract('ab,ib->ia', df_dR[v_, v_], t1)
                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t1)
                    dRt1_dR += oe.contract('jb,ijab->ia', df_dR[o_, v_], 2.0 * t2 - t2.swapaxes(2, 3))
                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_, o_, v_, v_] - dERI_dR.swapaxes(2, 3)[v_, o_, v_, v_], t2)
                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_, o_, o_, v_] - dERI_dR.swapaxes(2, 3)[o_, o_, o_, v_], t2)
                    dRt1_dR -= E_CISD * dt1_dR
                    dRt1_dR -= oe.contract('ji,ja->ia', F[o_, o_], dt1_dR)
                    dRt1_dR += oe.contract('ab,ib->ia', F[v_, v_], dt1_dR)
                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt1_dR)
                    dRt1_dR += oe.contract('jb,ijab->ia', F[o_, v_], 2.0 * dt2_dR - dt2_dR.swapaxes(2, 3))
                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_, o_, v_, v_] - ERI.swapaxes(2, 3)[v_, o_, v_, v_], dt2_dR)
                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_, o_, o_, v_] - ERI.swapaxes(2, 3)[o_, o_, o_, v_], dt2_dR)

                    dRt2_dR = dERI_dR.copy().swapaxes(0, 2).swapaxes(1, 3)[o_, o_, v_, v_]
                    dRt2_dR -= dE_dR_proj * t2
                    dRt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_, v_, v_, o_], t1)
                    dRt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_, v_, o_, v_], t1)
                    dRt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_, v_, o_, o_], t1)
                    dRt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_, o_, o_, o_], t1)
                    dRt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_, v_], t2)
                    dRt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_, v_], t2)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_, o_], t2)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_, o_], t2)
                    dRt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_, o_, o_, o_], t2)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_, v_, v_, v_], t2)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_, v_, v_, o_], t2)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_, v_, o_, v_], t2)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_, v_, v_, o_], t2)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_, v_, v_, o_] - dERI_dR.swapaxes(2, 3)[o_, v_, v_, o_], t2)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_, v_, o_, v_], t2)
                    dRt2_dR -= E_CISD * dt2_dR
                    dRt2_dR += oe.contract('abcj,ic->ijab', ERI[v_, v_, v_, o_], dt1_dR)
                    dRt2_dR += oe.contract('abic,jc->ijab', ERI[v_, v_, o_, v_], dt1_dR)
                    dRt2_dR -= oe.contract('kbij,ka->ijab', ERI[o_, v_, o_, o_], dt1_dR)
                    dRt2_dR -= oe.contract('akij,kb->ijab', ERI[v_, o_, o_, o_], dt1_dR)
                    dRt2_dR += oe.contract('ac,ijcb->ijab', F[v_, v_], dt2_dR)
                    dRt2_dR += oe.contract('bc,ijac->ijab', F[v_, v_], dt2_dR)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', F[o_, o_], dt2_dR)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', F[o_, o_], dt2_dR)
                    dRt2_dR += oe.contract('klij,klab->ijab', ERI[o_, o_, o_, o_], dt2_dR)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', ERI[v_, v_, v_, v_], dt2_dR)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', ERI[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', ERI[o_, v_, o_, v_], dt2_dR)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', ERI[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_, v_, v_, o_] - ERI.swapaxes(2, 3)[o_, v_, v_, o_], dt2_dR)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', ERI[o_, v_, o_, v_], dt2_dR)

                    dt1_dR += dRt1_dR / wfn_CISD.D_ia
                    dt2_dR += dRt2_dR / wfn_CISD.D_ijab

                    if self.parameters['DIIS']:
                        occ = len(dt1_dR)
                        vir = len(dt1_dR[0])
                        dt1_dR_flat = len(np.reshape(dt1_dR, (-1)))
                        dt2_dR_flat = len(np.reshape(dt2_dR, (-1)))
                        res_vec = np.concatenate((np.reshape(dRt1_dR, (-1)), np.reshape(dRt2_dR, (-1))))
                        t_vec = np.concatenate((np.reshape(dt1_dR, (-1)), np.reshape(dt2_dR, (-1))))
                        if iteration == 1:
                            t_iter = np.atleast_2d(t_vec).T
                            e_iter = np.atleast_2d(res_vec).T
                        t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                        dt1_dR = np.reshape(t_vec[0:dt1_dR_flat], (occ, vir))
                        dt2_dR = np.reshape(t_vec[dt1_dR_flat:], (occ, occ, vir, vir))

                    dE_dR_proj  = 2.0 * oe.contract('ia,ia->', t1, df_dR[o_, v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_, o_, v_, v_] - dERI_dR.swapaxes(2, 3)[o_, o_, v_, v_])
                    dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_, v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_, o_, v_, v_] - ERI.swapaxes(2, 3)[o_, o_, v_, v_])

                    rms_dt1_dR = np.sqrt(oe.contract('ia,ia->', dt1_dR_old - dt1_dR, dt1_dR_old - dt1_dR))
                    rms_dt2_dR = np.sqrt(oe.contract('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR))
                    delta_dE_dR_proj = dE_dR_proj_old - dE_dR_proj

                    if iteration > 1:
                        if abs(delta_dE_dR_proj) < self.parameters['e_convergence'] and rms_dt1_dR < self.parameters['d_convergence'] and rms_dt2_dR < self.parameters['d_convergence']:
                            break
                    if iteration == self.parameters['max_iterations']:
                        if abs(delta_dE_dR_proj) > self.parameters['e_convergence'] or rms_dt1_dR > self.parameters['d_convergence'] or rms_dt2_dR > self.parameters['d_convergence']:
                            print("Not converged.")
                    iteration += 1

                N_R = -(1 / np.sqrt((1 + 2 * oe.contract('ia,ia', np.conjugate(t1), t1) + oe.contract('ijab,ijab', np.conjugate(t2), 2 * t2 - t2.swapaxes(2, 3)))**3))
                N_R *= 0.5 * (2 * oe.contract('ia,ia', np.conjugate(dt1_dR), t1) + 2 * oe.contract('ia,ia', dt1_dR, np.conjugate(t1)) + oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2 * t2 - t2.swapaxes(2, 3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2 * t2 - t2.swapaxes(2, 3))))

                for beta in range(3):
                    lambda_alpha = 3 * N1 + a

                    if orbitals == 'canonical':
                        APT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        APT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_A[beta][v_, o_])
                        APT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_A[beta][v_, o_], U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        APT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_A[beta][v_, o], U_R[o_, o] + half_S_core_a[o, o_].T)

                        APT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ck", dT1_dA[beta], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fk", t1, U_A[beta][v_, v_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,cn", t1, U_A[beta][o_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dA[beta])
                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,cf,kf", dt1_dR, U_A[beta][v_, v_], t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,nk,nc", dt1_dR, U_A[beta][o_, o_], t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dA[beta], U_R[v_, v_] + half_S_core_a[v_, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dA[beta], U_R[o_, o_] + half_S_core_a[o_, o_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fa,ka", t1, U_A[beta][v_, v_], U_R[v_, v_] + half_S_core_a[v_, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fc,ik,if", t1, U_A[beta][v_, v_], U_R[o_, o_] + half_S_core_a[o_, o_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ca,na", t1, U_A[beta][o_, o_], U_R[v_, v_] + half_S_core_a[v_, v_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,kn,in,ic", t1, U_A[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,kc,ia,ia", t1, U_A[beta][o_, v_], U_R[o_, v_] + half_S_core_a[v_, o_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,fn,kc", t1, U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,fk,nc", t1, U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,cn,kf", t1, U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,ck,nf", t1, U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T, t1)

                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), U_A[beta][v_, o_], t1)
                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ia,ikac", dT1_dA[beta], U_R[o_, v_] + half_S_core_a[v_, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,ia,ikaf", t1, U_A[beta][v_, v_], U_R[o_, v_] + half_S_core_a[v_, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ia,inac", t1, U_A[beta][o_, o_], U_R[o_, v_] + half_S_core_a[v_, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,ik,incf", t1, U_A[beta][v_, o_], U_R[o_, o_] + half_S_core_a[o_, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,in,ikfc", t1, U_A[beta][v_, o], U_R[o_, o] + half_S_core_a[o, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,ca,knaf", t1, U_A[beta][v_, o_], U_R[v_, v_] + half_S_core_a[v_, v_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,fa,knca", t1, U_A[beta][v_, o_], U_R[v_, v_] + half_S_core_a[v_, v_].T, 2 * t2 - t2.swapaxes(2, 3))

                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_A[beta][o_, v_], 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,dl,kc", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), U_R[v_, o_] + half_S_core_a[o_, v_].T, t1)
                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ea,kice", t1, U_A[beta][o_, v_], U_R[v_, v_] + half_S_core_a[v_, v_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,kc,im,kmca", t1, U_A[beta][o_, v_], U_R[o_, o_] + half_S_core_a[o_, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ac,em,imce", t1, U_A[beta][v_, v_], U_R[v_, o_] + half_S_core_a[o_, v_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ec,em,imac", t1, U_A[beta][v_, v_], U_R[v_, o_] + half_S_core_a[o_, v_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,ki,em,kmae", t1, U_A[beta][o_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_A[beta][o_, o], U_R[v_, o] + half_S_core_a[o, v_].T, 2 * t2 - t2.swapaxes(2, 3))

                        APT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dA[beta])
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_A[beta][o_, o_])
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_A[beta][v_, v_])
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("klcd,mlcd,mk", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,kled,ce", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ec,ea", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, v_], U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        if normalization == 'full':
                            APT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("ijab,kjab,ki", 2 * t2 - t2.swapaxes(2, 3), t2, U_A[beta][o_, o_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,ijcb,ac", 2 * t2 - t2.swapaxes(2, 3), t2, U_A[beta][v_, v_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dA[beta])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ai", t1, U_A[beta][v_, o_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("kc,kc", t1, U_A[beta][o_, v_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dA[beta])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("kc,cf,kf", t1, U_A[beta][v_, v_], t1)
                            APT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("kc,nk,nc", t1, U_A[beta][o_, o_], t1)
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,bj,ia", 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], t1)
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,kc,ikac", t1, U_A[beta][o_, v_], 2 * t2 - t2.swapaxes(2, 3))

                    if orbitals == 'non-canonical':
                        APT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        APT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_A[beta][v_, o_])
                        APT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_A[beta][v_, o_], U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        APT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_A[beta][v_, o], U_R[o_, o] + half_S_core_a[o, o_].T)

                        APT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dT1_dA[beta], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,im,am", t1, U_A[beta][o_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dA[beta])
                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dA[beta], U_R[v_, v_] + half_S_core_a[v_, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dA[beta], U_R[o_, o_] + half_S_core_a[o_, o_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,im,km,ka", t1, U_A[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,ia,kc,kc", t1, U_A[beta][o_, v_], U_R[o_, v_] + half_S_core_a[v_, o_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,em,ia", t1, U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,ei,ma", t1, U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T, t1)
                        APT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,am,ie", t1, U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T, t1)
                        APT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,ai,me", t1, U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T, t1)

                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), U_A[beta][v_, o_], t1)
                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dT1_dA[beta], U_R[o_, v_] + half_S_core_a[v_, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,km,kiea", t1, U_A[beta][v_, o], U_R[o_, o] + half_S_core_a[o, o_].T, 2 * t2 - t2.swapaxes(2, 3))
                        APT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,em,ec,imac", t1, U_A[beta][v_, o_], U_R[v_, v_] + half_S_core_a[v_, v_].T, 2 * t2 - t2.swapaxes(2, 3))

                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_A[beta][o_, v_], 2 * t2 - t2.swapaxes(2, 3))
                        APT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), U_R[v_, o_] + half_S_core_a[o_, v_].T, t1)
                        APT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_A[beta][o_, o], U_R[v_, o] + half_S_core_a[o, v_].T, 2 * t2 - t2.swapaxes(2, 3))

                        APT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dA[beta])
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2 * dT2_dA[beta] - dT2_dA[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        APT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        APT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        if normalization == 'full':
                            APT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dA[beta])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ia,ai", t1, U_A[beta][v_, o_])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dA[beta])
                            APT_Norm[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ijab,bj,ia", 2 * t2 - t2.swapaxes(2, 3), U_A[beta][v_, o_], t1)

        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()
        Nuc = np.zeros((3 * natom, 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * natom):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                Nuc[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        APT_total = APT_HF + APT_S0 + APT_0S + APT_SS + APT_DS + APT_SD + APT_DD + APT_Norm
        return -2 * APT_total + Nuc
