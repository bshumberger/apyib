"""This script contains a set of functions for analytic evaluation of the atomic polar tensors."""

import numpy as np
import opt_einsum as oe
from apyib.analytic_base import AnalyticDerivative


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
