"""This script contains a set of functions for analytic evaluation of the atomic axial tensors."""

import numpy as np
import psi4
import gc
import opt_einsum as oe
from apyib.analytic_base import AnalyticDerivative
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn
from apyib.utils import solve_general_DIIS


class analytic_derivative(AnalyticDerivative):
    """Analytic atomic axial tensors for RHF, MP2, CID, and CISD wavefunctions."""



    def compute_RHF_AATs(self, orbitals='non-canonical'):
        """Compute analytic RHF atomic axial tensors.

        Parameters
        ----------
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).

        Returns
        -------
        AAT_HF : ndarray, shape (3*natom, 3)
            Hartree-Fock contribution to the AAT ``dI_beta / dR_alpha`` [a.u.].
        """
        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        npert_nuc = 3 * natom
        B_nuc = np.zeros((nv * no, npert_nuc))
        S_nuc = np.zeros((npert_nuc, nbf, nbf))
        h_dep_nuc = np.zeros((npert_nuc, nbf, nbf))
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

        # Nuclear CPHF: real perturbation, ov_sign=-1, dep_sign=+1.
        U_R = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        A_mag, G_mag = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        mu_mag_AO = mints.ao_angular_momentum()
        B_mag = np.zeros((nv * no, 3))
        h_dep_mag = np.zeros((3, nbf, nbf))
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)
            B_mag[:, a] = mu_mag[v, o].reshape(nv * no)
            h_dep_mag[a] = mu_mag

        # Magnetic CPHF: imaginary perturbation, ov_sign=+1, dep_sign=-1.
        U_H = self._solve_cphf(G_mag, A_mag, B_mag, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=+1, h_dep_all=h_dep_mag, dep_sign=-1)

        AAT_HF = np.zeros((natom * 3, 3))
        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                AAT_HF[lambda_alpha][beta] += 2 * oe.contract("em,em", U_H[beta][v_, o],
                                                               U_R[lambda_alpha][v_, o]
                                                               + half_S[lambda_alpha][o, v_].T)

        return AAT_HF



    def compute_MP2_AATs(self, normalization='full', orbitals='non-canonical'):
        """Compute analytic MP2 atomic axial tensors.

        Parameters
        ----------
        normalization : {'full', 'intermediate'}, optional
            Wavefunction normalization convention (default ``'full'``).
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).

        Returns
        -------
        ndarray, shape (3*natom, 3)
            MP2 AAT ``dI_beta / dR_alpha`` [a.u.].
        """
        wfn_MP2 = mp2_wfn(self.parameters, self.wfn)
        E_MP2, t2 = wfn_MP2.solve_MP2()

        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints

        AAT_HF = np.zeros((natom * 3, 3))
        AAT_1 = np.zeros((natom * 3, 3))
        AAT_2 = np.zeros((natom * 3, 3))
        AAT_3 = np.zeros((natom * 3, 3))
        AAT_4 = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + oe.contract('ijab,ijab', t2, 2 * t2 - t2.swapaxes(2, 3)))

        # ------------------------------------------------------------------ #
        # Magnetic CPHF: vectorized over 3 field directions.                 #
        # ------------------------------------------------------------------ #
        A_mag, G_mag = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        mu_mag_AO = mints.ao_angular_momentum()
        B_mag = np.zeros((nv * no, 3))
        h_dep_mag = np.zeros((3, nbf, nbf))
        for b in range(3):
            mu_mag_AO[b] = -0.5 * mu_mag_AO[b].np
            mu_mag = oe.contract('mp,mn,nq->pq', C, mu_mag_AO[b], C)
            B_mag[:, b] = mu_mag[v, o].reshape(nv * no)
            h_dep_mag[b] = mu_mag

        # Magnetic CPHF: imaginary perturbation, ov_sign=+1, dep_sign=-1.
        U_H = self._solve_cphf(G_mag, A_mag, B_mag, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=+1, h_dep_all=h_dep_mag, dep_sign=-1)

        # Post-CPHF magnetic processing: df_dH, dERI_dH, dt2_dH per direction.
        dT2_dH = []
        for b in range(3):
            U_h = U_H[b]
            h_core = h_dep_mag[b]

            # Derivative of the MO Fock matrix w.r.t. the magnetic field, occ-occ
            # and virt-virt blocks: perturbation operator + orbital-energy-weighted
            # U_H response + A_mag coupling through the solved U_H[v,o].
            df_dH = np.zeros((nbf, nbf))
            df_dH[o, o] -= h_core[o, o].copy()
            df_dH[o, o] += U_h[o, o] * self.wfn.eps[o].reshape(-1, 1) - U_h[o, o].swapaxes(0, 1) * self.wfn.eps[o]
            df_dH[o, o] += oe.contract('em,iejm->ij', U_h[v, o], A_mag.swapaxes(1, 2)[o, v, o, o])
            df_dH[v, v] -= h_core[v, v].copy()
            df_dH[v, v] += U_h[v, v] * self.wfn.eps[v].reshape(-1, 1) - U_h[v, v].swapaxes(0, 1) * self.wfn.eps[v]
            df_dH[v, v] += oe.contract('em,aebm->ab', U_h[v, o], A_mag.swapaxes(1, 2)[v, v, v, o])

            # Derivative of the MO two-electron integrals: U_H orbital rotation
            # applied to each of the four indices in turn.
            dERI_dH  = oe.contract('tr,pqts->pqrs', U_h[:, t], ERI[t, t, :, t])
            dERI_dH += oe.contract('ts,pqrt->pqrs', U_h[:, t], ERI[t, t, t, :])
            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_h[:, t], ERI[:, t, t, t])
            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_h[:, t], ERI[t, :, t, t])

            # Derivative of the MP2 T2 amplitudes: differentiated integrals plus
            # Fock-derivative terms, divided by the MP2 energy denominator.
            dt2_dH = dERI_dH.copy().swapaxes(0, 2).swapaxes(1, 3)[o_, o_, v_, v_]
            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_, v_], t2)
            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_, v_], t2)
            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_, o_], t2)
            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_, o_], t2)
            dt2_dH /= wfn_MP2.D_ijab

            dT2_dH.append(dt2_dH)
            print("\nMagnetic Field Perturbtion Data:")
            print("Cartesian: ", b)
            print("Maximum dt2/dH: ", np.max(dt2_dH))

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

                # Skeleton-derivative Fock matrix and CPHF right-hand side for
                # nuclear perturbation k = 3*N1 + a (cf. analytic_hessian).
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

        # Nuclear CPHF: real perturbation, ov_sign=-1, dep_sign=+1.
        U_R_list = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                     ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        # Post-CPHF nuclear processing: df_dR, dERI_dR, dt2_dR, AAT contractions.
        for N1 in atoms:
            for a in range(3):
                k = 3 * N1 + a
                U_R = U_R_list[k]
                F_core = h_dep_nuc[k]
                S_core_a = S_nuc[k]
                ERI_core_a = ERI_core_nuc[k]
                half_S_core_a = half_S[k]
                lambda_alpha = k

                # Derivative of the MO Fock matrix w.r.t. nuclear displacement
                # (occ-occ, virt-virt): skeleton Fock + U_R response + A coupling +
                # overlap-derivative correction.
                df_dR = np.zeros((nbf, nbf))
                df_dR[o, o] += F_core[o, o].copy()
                df_dR[o, o] += U_R[o, o] * self.wfn.eps[o].reshape(-1, 1) + U_R[o, o].swapaxes(0, 1) * self.wfn.eps[o]
                df_dR[o, o] += oe.contract('em,iejm->ij', U_R[v, o], A.swapaxes(1, 2)[o, v, o, o])
                df_dR[o, o] -= 0.5 * oe.contract('mn,imjn->ij', S_core_a[o, o], A.swapaxes(1, 2)[o, o, o, o])
                df_dR[v, v] += F_core[v, v].copy()
                df_dR[v, v] += U_R[v, v] * self.wfn.eps[v].reshape(-1, 1) + U_R[v, v].swapaxes(0, 1) * self.wfn.eps[v]
                df_dR[v, v] += oe.contract('em,aebm->ab', U_R[v, o], A.swapaxes(1, 2)[v, v, v, o])
                df_dR[v, v] -= 0.5 * oe.contract('mn,ambn->ab', S_core_a[o, o], A.swapaxes(1, 2)[v, o, v, o])

                # Derivative of the MO two-electron integrals: skeleton derivative
                # plus the U_R rotation applied to each of the four indices.
                dERI_dR = ERI_core_a.copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:, t], ERI[:, t, t, t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:, t], ERI[t, :, t, t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:, t], ERI[t, t, :, t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:, t], ERI[t, t, t, :])

                # Derivative of the MP2 T2 amplitudes w.r.t. nuclear displacement.
                dt2_dR = dERI_dR.copy()[o_, o_, v_, v_]
                dt2_dR -= oe.contract('kjab,ik->ijab', t2, df_dR[o_, o_])
                dt2_dR -= oe.contract('ikab,kj->ijab', t2, df_dR[o_, o_])
                dt2_dR += oe.contract('ijcb,ac->ijab', t2, df_dR[v_, v_])
                dt2_dR += oe.contract('ijac,cb->ijab', t2, df_dR[v_, v_])
                dt2_dR /= wfn_MP2.D_ijab

                print("\nNuclear Perturbation Data:")
                print("Atom: ", N1)
                print("Cartesian: ", a)
                print("Maximum dt2/dR: ", np.max(dt2_dR))

                # Derivative of the full-normalization factor w.r.t. this displacement.
                N_R = -(1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2 * t2 - t2.swapaxes(2, 3)))**3))
                N_R *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2 * t2 - t2.swapaxes(2, 3))
                              + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2 * t2 - t2.swapaxes(2, 3))))

                # AAT components, summed at the end: AAT_HF (reference/reference),
                # AAT_1-4 (doubles/doubles), AAT_Norm (normalization-derivative
                # correction, full normalization only). The canonical and
                # non-canonical branches differ in the within-block U treatment.
                for beta in range(3):
                    if orbitals == 'canonical':
                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        AAT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dH[beta])

                        AAT_2[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_H[beta][o_, o_])
                        AAT_2[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), t2, U_H[beta][v_, v_])

                        AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("klcd,mlcd,mk", 2 * dT2_dH[beta] - dT2_dH[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        AAT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("klcd,kled,ce", 2 * dT2_dH[beta] - dT2_dH[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)

                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ec,ea", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core_a[v_, v_].T)
                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        if normalization == 'full':
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2.0 * oe.contract("ijab,kjab,ki", 2 * t2 - t2.swapaxes(2, 3), t2, U_H[beta][o_, o_])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2.0 * oe.contract("ijab,ijcb,ac", 2 * t2 - t2.swapaxes(2, 3), t2, U_H[beta][v_, v_])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dH[beta])

                    if orbitals == 'non-canonical':
                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        AAT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2 * dt2_dR - dt2_dR.swapaxes(2, 3), dT2_dH[beta])

                        AAT_2[lambda_alpha][beta] += 0

                        AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2 * dT2_dH[beta] - dT2_dH[beta].swapaxes(2, 3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        AAT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2 * dT2_dH[beta] - dT2_dH[beta].swapaxes(2, 3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)

                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2 * t2 - t2.swapaxes(2, 3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        if normalization == 'full':
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * oe.contract("ijab,ijab", 2 * t2 - t2.swapaxes(2, 3), dT2_dH[beta])

        print("\nHartree-Fock AAT:")
        print(AAT_HF, "\n")
        print("Doubles/Doubles:")
        print(AAT_1 + AAT_2 + AAT_3 + AAT_4, "\n")

        return AAT_HF + AAT_1 + AAT_2 + AAT_3 + AAT_4 + AAT_Norm



    def compute_CISD_AATs(self, normalization='full', orbitals='non-canonical', print_level=0):
        """Compute analytic CISD atomic axial tensors.

        Parameters
        ----------
        normalization : {'full', 'intermediate'}, optional
            Wavefunction normalization convention (default ``'full'``).
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).
        print_level : int, optional
            Verbosity level; 0 (default) suppresses output.

        Returns
        -------
        ndarray, shape (3*natom, 3)
            CISD AAT ``dI_beta / dR_alpha`` [a.u.].
        """
        # Compute T2 amplitudes and MP2 energy.
        wfn_CISD = ci_wfn(self.parameters, self.wfn)
        E_CISD, t1, t2 = wfn_CISD.solve_CISD()

        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Setting up different components of the AATs.
        # CISD AAT components labelled by bra/ket excitation level: the first
        # label is the nuclear-derivative (bra) block and the second the
        # magnetic-derivative (ket) block, with 0 = reference, S = singles,
        # D = doubles. AAT_Norm is the full-normalization derivative correction.
        AAT_HF = np.zeros((natom * 3, 3))
        AAT_S0 = np.zeros((natom * 3, 3))
        AAT_0S = np.zeros((natom * 3, 3))
        AAT_SS = np.zeros((natom * 3, 3))
        AAT_DS = np.zeros((natom * 3, 3))
        AAT_SD = np.zeros((natom * 3, 3))
        AAT_DD = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        ########## Test ############
        ## Setting up different components of the AATs.
        #AAT_HF_test = np.zeros((natom * 3, 3))
        #AAT_S0_test = np.zeros((natom * 3, 3))
        #AAT_0S_test = np.zeros((natom * 3, 3))
        #AAT_SS_test = np.zeros((natom * 3, 3))
        #AAT_DS_test = np.zeros((natom * 3, 3))
        #AAT_SD_test = np.zeros((natom * 3, 3))
        #AAT_DD_test = np.zeros((natom * 3, 3))
        #AAT_Norm_test = np.zeros((natom * 3, 3))
        ########### End ############

        # Compute normalization factor.
        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + 2*oe.contract('ia,ia', t1, t1) + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        # Set up derivative t-amplitude matrices.
        dT1_dH = []
        dT2_dH = []

        # Set up U-coefficient matrices for AAT calculations.
        U_H = []

        # Compute OPD and TPD matrices for use in computing the energy gradient.
        # Compute normalize amplitudes.
        N = 1 / np.sqrt(1**2 + 2*oe.contract('ia,ia->', np.conjugate(t1), t1) + oe.contract('ijab,ijab->', np.conjugate(t2), 2*t2-t2.swapaxes(2,3)))
        t0_n = N.copy()
        t1_n = t1 * N
        t2_n = t2 * N

        # Build the CISD one-particle density matrix (OPD) in the MO basis from
        # the normalized reference/singles/doubles amplitudes (t0_n, t1_n, t2_n).
        D_pq = np.zeros_like(F)
        D_pq[o_,o_] -= 2 * oe.contract('ja,ia->ij', np.conjugate(t1_n), t1_n) + 2 * oe.contract('jkab,ikab->ij', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[v_,v_] += 2 * oe.contract('ia,ib->ab', np.conjugate(t1_n), t1_n) + 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[o_,v_] += 2 * np.conjugate(t0_n) * t1_n + 2 * oe.contract('jb,ijab->ia', np.conjugate(t1_n), t2_n - t2_n.swapaxes(2,3))
        D_pq[v_,o_] += 2 * np.conjugate(t1_n.T) * t0_n + 2 * oe.contract('ijab,jb->ai', np.conjugate(t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pq = D_pq[t_,t_]

        # Build the CISD two-particle density matrix (TPD) in the MO basis.
        D_pqrs = np.zeros_like(ERI)
        D_pqrs[o_,o_,o_,o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[v_,v_,v_,v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[o_,v_,v_,o_] += 4 * oe.contract('ja,ib->iabj', np.conjugate(t1_n), t1_n)
        D_pqrs[o_,v_,o_,v_] -= 2 * oe.contract('ja,ib->iajb', np.conjugate(t1_n), t1_n)
        D_pqrs[v_,o_,o_,v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), 2*t2_n - t2_n.swapaxes(2,3))

        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)

        D_pqrs[o_,o_,v_,v_] += np.conjugate(t0_n) * (2*t2_n -t2_n.swapaxes(2,3))
        D_pqrs[v_,v_,o_,o_] += np.conjugate(2*t2_n.swapaxes(0,2).swapaxes(1,3) - t2_n.swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)) * t0_n
        D_pqrs[v_,o_,v_,v_] += 2 * oe.contract('ja,ijcb->aibc', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
        D_pqrs[o_,v_,o_,o_] -= 2 * oe.contract('kjab,ib->iajk', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pqrs[v_,v_,v_,o_] += 2 * oe.contract('jiab,jc->abci', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pqrs[o_,o_,o_,v_] -= 2 * oe.contract('kb,ijba->ijka', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
        D_pqrs = D_pqrs[t_,t_,t_,t_]

        A_mag, G_mag = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        # Collect magnetic B vectors and perturbation operators for vectorized CPHF solve.
        mu_mag_AO = mints.ao_angular_momentum()
        B_mag = np.zeros((nv * no, 3))
        h_dep_mag = np.zeros((3, nbf, nbf))
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)
            B_mag[:, a] = mu_mag[v, o].reshape(nv * no)
            h_dep_mag[a] = mu_mag

        # Vectorized magnetic CPHF solve (all 3 directions at once).
        U_H = self._solve_cphf(G_mag, A_mag, B_mag, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                ov_sign=+1, h_dep_all=h_dep_mag, dep_sign=-1)

        # Per-direction DIIS post-processing for magnetic amplitude derivatives.
        for a in range(3):
            h_core = h_dep_mag[a]
            U_h = U_H[a]

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            df_dH = np.zeros((nbf,nbf))

            df_dH[o,o] -= h_core[o,o].copy()
            df_dH[o,o] += U_h[o,o] * self.wfn.eps[o].reshape(-1,1) - U_h[o,o].swapaxes(0,1) * self.wfn.eps[o]
            df_dH[o,o] += oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            df_dH[v,v] -= h_core[v,v].copy()
            df_dH[v,v] += U_h[v,v] * self.wfn.eps[v].reshape(-1,1) - U_h[v,v].swapaxes(0,1) * self.wfn.eps[v]
            df_dH[v,v] += oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            dERI_dH =  oe.contract('tr,pqts->pqrs', U_h[:,t], ERI[t,t,:,t])
            dERI_dH += oe.contract('ts,pqrt->pqrs', U_h[:,t], ERI[t,t,t,:])
            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_h[:,t], ERI[:,t,t,t])
            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_h[:,t], ERI[t,:,t,t])

            # Compute CISD energy gradient.
            dE_dH = oe.contract('pq,pq->', df_dH[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dH[t_,t_,t_,t_], D_pqrs)

            # Computing the HF energy gradient.
            dE_dH_HF = 2 * oe.contract('ii->', h_core[o,o])
            dE_dH_tot = dE_dH + dE_dH_HF

            # Compute dT1_dH guess amplitudes.
            dt1_dH = -dE_dH * t1
            dt1_dH -= oe.contract('ji,ja->ia', df_dH[o_,o_], t1)
            dt1_dH += oe.contract('ab,ib->ia', df_dH[v_,v_], t1)
            dt1_dH += oe.contract('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
            dt1_dH += oe.contract('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
            dt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
            dt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)
            dt1_dH /= wfn_CISD.D_ia

            # Compute dT2_dH guess amplitudes.
            dt2_dH = -dE_dH * t2
            dt2_dH += oe.contract('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
            dt2_dH += oe.contract('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
            dt2_dH -= oe.contract('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
            dt2_dH -= oe.contract('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
            dt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
            dt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
            dt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH /= wfn_CISD.D_ijab

            # Solve for initial CISD energy gradient.
            dE_dH_proj =  2.0 * oe.contract('ia,ia->', t1, df_dH[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
            dE_dH_proj += 2.0 * oe.contract('ia,ia->', dt1_dH, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
            dt1_dH = dt1_dH.copy()
            dt2_dH = dt2_dH.copy()

            # Start iterative procedure.
            iteration = 1
            while iteration <= self.parameters['max_iterations']:
                dE_dH_proj_old = dE_dH_proj
                dt1_dH_old = dt1_dH.copy()
                dt2_dH_old = dt2_dH.copy()

                # Solving for the derivative residuals.
                dRt1_dH = df_dH.copy().swapaxes(0,1)[o_,v_]

                dRt1_dH -= dE_dH_proj * t1
                dRt1_dH -= oe.contract('ji,ja->ia', df_dH[o_,o_], t1)
                dRt1_dH += oe.contract('ab,ib->ia', df_dH[v_,v_], t1)
                dRt1_dH += oe.contract('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
                dRt1_dH += oe.contract('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                dRt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
                dRt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)

                dRt1_dH -= E_CISD * dt1_dH
                dRt1_dH -= oe.contract('ji,ja->ia', F[o_,o_], dt1_dH)
                dRt1_dH += oe.contract('ab,ib->ia', F[v_,v_], dt1_dH)
                dRt1_dH += oe.contract('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dH)
                dRt1_dH += oe.contract('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dH - dt2_dH.swapaxes(2,3))
                dRt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dH)
                dRt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dH)

                dRt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                dRt2_dH -= dE_dH_proj * t2
                dRt2_dH += oe.contract('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
                dRt2_dH += oe.contract('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
                dRt2_dH -= oe.contract('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
                dRt2_dH -= oe.contract('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
                dRt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
                dRt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
                dRt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
                dRt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
                dRt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
                dRt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
                dRt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
                dRt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)

                dRt2_dH -= E_CISD * dt2_dH
                dRt2_dH += oe.contract('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dH)
                dRt2_dH += oe.contract('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dH)
                dRt2_dH -= oe.contract('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dH)
                dRt2_dH -= oe.contract('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dH)
                dRt2_dH += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dH)
                dRt2_dH += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dH)
                dRt2_dH -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dH)
                dRt2_dH += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dH)
                dRt2_dH -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dH)
                dRt2_dH -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dH)

                dt1_dH += dRt1_dH / wfn_CISD.D_ia
                dt2_dH += dRt2_dH / wfn_CISD.D_ijab

                # Perform DIIS extrapolation.
                if self.parameters['DIIS']:
                    occ = len(dt1_dH)
                    vir = len(dt1_dH[0])
                    dt1_dH_flat = len(np.reshape(dt1_dH, (-1)))
                    dt2_dH_flat = len(np.reshape(dt2_dH, (-1)))
                    res_vec = np.concatenate((np.reshape(dRt1_dH, (-1)), np.reshape(dRt2_dH, (-1))))
                    t_vec = np.concatenate((np.reshape(dt1_dH, (-1)), np.reshape(dt2_dH, (-1))))
                    if iteration == 1:
                        t_iter = np.atleast_2d(t_vec).T
                        e_iter = np.atleast_2d(res_vec).T
                    t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                    dt1_dH = np.reshape(t_vec[0:dt1_dH_flat], (occ, vir))
                    dt2_dH = np.reshape(t_vec[dt1_dH_flat:], (occ, occ, vir, vir))

                # Compute new CISD energy gradient.
                dE_dH_proj =  2.0 * oe.contract('ia,ia->', t1, df_dH[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dH_proj += 2.0 * oe.contract('ia,ia->', dt1_dH, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                # Compute new total energy gradient.
                dE_dH_tot_proj = dE_dH_proj + dE_dH_HF

                # Compute convergence data.
                rms_dt1_dH = oe.contract('ia,ia->', dt1_dH_old - dt1_dH, dt1_dH_old - dt1_dH)
                rms_dt1_dH = np.sqrt(rms_dt1_dH)

                rms_dt2_dH = oe.contract('ijab,ijab->', dt2_dH_old - dt2_dH, dt2_dH_old - dt2_dH)
                rms_dt2_dH = np.sqrt(rms_dt2_dH)
                delta_dE_dH_proj = dE_dH_proj_old - dE_dH_proj

                if print_level > 0:
                    print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dH_proj, dE_dH_tot_proj, delta_dE_dH_proj, rms_dt1_dH, rms_dt2_dH))

                if iteration > 1:
                    if abs(delta_dE_dH_proj) < self.parameters['e_convergence'] and rms_dt1_dH < self.parameters['d_convergence'] and rms_dt2_dH < self.parameters['d_convergence']:
                        #print("Convergence criteria met.")
                        break
                if iteration == self.parameters['max_iterations']:
                    if abs(delta_dE_dH_proj) > self.parameters['e_convergence'] or rms_dt1_dH > self.parameters['d_convergence'] or rms_dt2_dH > self.parameters['d_convergence']:
                        print("Not converged.")
                iteration += 1

            print("\nMagnetic Field Perturbtion Data:")
            print("Cartesian: ", a)
            print("Maximum dt1/dH: ", np.max(dt1_dH))
            print("Maximum dt2/dH: ", np.max(dt2_dH))

            dT1_dH.append(dt1_dH)
            dT2_dH.append(dt2_dH)


        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        # First pass: collect nuclear perturbation integrals and B vectors.
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
                F_core = h_core + oe.contract('piqi->pq', 2 * ERI_arr[a][:,o,:,o] - ERI_arr[a].swapaxes(2,3)[:,o,:,o])

                B_nuc[:, k] = (-F_core[v,o]
                               + oe.contract('ai,ii->ai', S_arr[a][v,o], F[o,o])
                               + 0.5 * oe.contract('mn,amin->ai', S_arr[a][o,o], A.swapaxes(1,2)[v,o,o,o])
                               ).reshape(nv * no)
                S_nuc[k] = S_arr[a]
                h_dep_nuc[k] = F_core

                h_core_store.append(h_core)
                ERI_core_store.append(ERI_arr[a].copy())
                half_S_core_store.append(half_S_arr[a])

        # Vectorized nuclear CPHF solve (all 3*natom directions at once).
        U_R_list = self._solve_cphf(G, A, B_nuc, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                                     ov_sign=-1, S_all=S_nuc, h_dep_all=h_dep_nuc, dep_sign=+1)

        # Second pass: DIIS for amplitude derivatives and AAT contractions.
        for N1 in atoms:
            for a in range(3):
                k = 3 * N1 + a
                U_R = U_R_list[k]
                h_core = h_core_store[k]
                F_core = h_dep_nuc[k]
                S_core_a = S_nuc[k]
                ERI_core_a = ERI_core_store[k]
                half_S_core_a = half_S_core_store[k]

                # Computing the gradient of the Fock matrix.
                df_dR = np.zeros((nbf,nbf))

                df_dR[o,o] += F_core[o,o].copy()
                df_dR[o,o] += U_R[o,o] * self.wfn.eps[o].reshape(-1,1) + U_R[o,o].swapaxes(0,1) * self.wfn.eps[o]
                df_dR[o,o] += oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o])
                df_dR[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_core_a[o,o], A.swapaxes(1,2)[o,o,o,o])

                df_dR[v,v] += F_core[v,v].copy()
                df_dR[v,v] += U_R[v,v] * self.wfn.eps[v].reshape(-1,1) + U_R[v,v].swapaxes(0,1) * self.wfn.eps[v]
                df_dR[v,v] += oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o])
                df_dR[v,v] -= 0.5 * oe.contract('mn,ambn->ab', S_core_a[o,o], A.swapaxes(1,2)[v,o,v,o])

                # Computing the gradient of the ERIs.
                dERI_dR = ERI_core_a.copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:,t], ERI[:,t,t,t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:,t], ERI[t,:,t,t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:,t], ERI[t,t,:,t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:,t], ERI[t,t,t,:])

                # Compute CISD energy gradient.
                dE_dR = oe.contract('pq,pq->', df_dR[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dR[t_,t_,t_,t_], D_pqrs)

                # Computing the HF energy gradient.
                dE_dR_HF = 2 * oe.contract('ii->', h_core[o,o])
                dE_dR_HF += oe.contract('ijij->', 2 * ERI_core_a[o,o,o,o] - ERI_core_a.swapaxes(2,3)[o,o,o,o])
                dE_dR_HF -= 2 * oe.contract('ii,i->', S_core_a[o,o], self.wfn.eps[o])
                dE_dR_HF += Nuc_Gradient[N1][a]

                dE_dR_tot = dE_dR + dE_dR_HF

                # Compute dT1_dR guess amplitudes.
                dt1_dR = -dE_dR * t1
                dt1_dR -= oe.contract('ji,ja->ia', df_dR[o_,o_], t1)
                dt1_dR += oe.contract('ab,ib->ia', df_dR[v_,v_], t1)
                dt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1) 
                dt1_dR += oe.contract('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                dt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
                dt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)
                dt1_dR /= wfn_CISD.D_ia

                # Compute dT2_dR guess amplitudes.
                dt2_dR = -dE_dR * t2
                dt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1) 
                dt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1) 
                dt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1) 
                dt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1) 
                dt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2) 
                dt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2) 
                dt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2) 
                dt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2) 
                dt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2) 
                dt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)    
                dt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2) 
                dt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
                dt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                dt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
                dt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR /= wfn_CISD.D_ijab

                # Solve for initial CISD energy gradient.
                dE_dR_proj =  2.0 * oe.contract('ia,ia->', t1, df_dR[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
                dt1_dR = dt1_dR.copy()
                dt2_dR = dt2_dR.copy()                

                # Start iterative procedure.
                iteration = 1
                while iteration <= self.parameters['max_iterations']:
                    dE_dR_proj_old = dE_dR_proj
                    dt1_dR_old = dt1_dR.copy()
                    dt2_dR_old = dt2_dR.copy()

                    # Solving for the derivative residuals.
                    dRt1_dR = df_dR.copy().swapaxes(0,1)[o_,v_]

                    dRt1_dR -= dE_dR_proj * t1
                    dRt1_dR -= oe.contract('ji,ja->ia', df_dR[o_,o_], t1)
                    dRt1_dR += oe.contract('ab,ib->ia', df_dR[v_,v_], t1)
                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1)
                    dRt1_dR += oe.contract('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)

                    dRt1_dR -= E_CISD * dt1_dR
                    dRt1_dR -= oe.contract('ji,ja->ia', F[o_,o_], dt1_dR)
                    dRt1_dR += oe.contract('ab,ib->ia', F[v_,v_], dt1_dR)
                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dR)
                    dRt1_dR += oe.contract('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dR - dt2_dR.swapaxes(2,3))
                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dR)
                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dR)

                    dRt2_dR = dERI_dR.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                    dRt2_dR -= dE_dR_proj * t2
                    dRt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1)
                    dRt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1)
                    dRt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1)
                    dRt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1)
                    dRt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2)
                    dRt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)

                    dRt2_dR -= E_CISD * dt2_dR
                    dRt2_dR += oe.contract('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dR)
                    dRt2_dR += oe.contract('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dR)
                    dRt2_dR -= oe.contract('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dR)
                    dRt2_dR -= oe.contract('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dR)
                    dRt2_dR += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dR)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dR)

                    dt1_dR += dRt1_dR / wfn_CISD.D_ia
                    dt2_dR += dRt2_dR / wfn_CISD.D_ijab

                    # Perform DIIS extrapolation.
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

                    # Compute new CISD energy gradient.
                    dE_dR_proj =  2.0 * oe.contract('ia,ia->', t1, df_dR[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                    dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                    # Compute new total energy gradient.
                    dE_dR_tot_proj = dE_dR_proj + dE_dR_HF

                    # Compute convergence data.
                    rms_dt1_dR = oe.contract('ia,ia->', dt1_dR_old - dt1_dR, dt1_dR_old - dt1_dR) 
                    rms_dt1_dR = np.sqrt(rms_dt1_dR)

                    rms_dt2_dR = oe.contract('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR) 
                    rms_dt2_dR = np.sqrt(rms_dt2_dR)
                    delta_dE_dR_proj = dE_dR_proj_old - dE_dR_proj

                    if print_level > 0:
                        print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dR_proj, dE_dR_tot_proj, delta_dE_dR_proj, rms_dt1_dR, rms_dt2_dR))

                    if iteration > 1:
                        if abs(delta_dE_dR_proj) < self.parameters['e_convergence'] and rms_dt1_dR < self.parameters['d_convergence'] and rms_dt2_dR < self.parameters['d_convergence']:
                            #print("Convergence criteria met.")
                            break
                    if iteration == self.parameters['max_iterations']:
                        if abs(delta_dE_dR_proj) > self.parameters['e_convergence'] or rms_dt1_dR > self.parameters['d_convergence'] or rms_dt2_dR > self.parameters['d_convergence']:
                            print("Not converged.")
                    iteration += 1

                print("\nNuclear Perturbation Data:")
                print("Atom: ", N1)
                print("Cartesian: ", a)
                print("Maximum dt1/dR: ", np.max(dt1_dR))
                print("Maximum dt2/dR: ", np.max(dt2_dR))

                # Compute derivative of the normalization factor.
                N_R = - (1 / np.sqrt((1 + 2*oe.contract('ia,ia', np.conjugate(t1), t1) + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_R *= 0.5 * (2*oe.contract('ia,ia', np.conjugate(dt1_dR), t1) + 2*oe.contract('ia,ia', dt1_dR, np.conjugate(t1)) + oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))

                for beta in range(0,3):
                    #Setting up AAT indexing.
                    lambda_alpha = 3 * N1 + a

                    if orbitals == 'canonical':
                        # Computing the Hartree-Fock term of the AAT.
                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        # Singles/Refence terms.
                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_H[beta][v_,o_])

                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core_a[v_,v_].T)
                        AAT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core_a[o,o_].T)

                        # Reference/Singles terms.
                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ck", dT1_dH[beta], U_R[v_,o_] + half_S_core_a[o_,v_].T)

                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fk", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core_a[o_,v_].T) # Canonical
                        AAT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,cn", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core_a[o,v_].T)                

                        # Singles/Singles terms.
                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dH[beta])

                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,cf,kf", dt1_dR, U_H[beta][v_,v_], t1) # Canonical
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,nk,nc", dt1_dR, U_H[beta][o_,o_], t1) # Canonical

                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], U_R[v_,v_] + half_S_core_a[v_,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], U_R[o_,o_] + half_S_core_a[o_,o_].T, t1)

                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fa,ka", t1, U_H[beta][v_,v_], U_R[v_,v_] + half_S_core_a[v_,v_].T, t1) # Canonical
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fc,ik,if", t1, U_H[beta][v_,v_], U_R[o_,o_] + half_S_core_a[o_,o_].T, t1) # Canonical    # TPD Contributor - Remove
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ca,na", t1, U_H[beta][o_,o_], U_R[v_,v_] + half_S_core_a[v_,v_].T, t1) # Canonical    # TPD Contributor - Remove
                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,kn,in,ic", t1, U_H[beta][o_,o], U_R[o_,o] + half_S_core_a[o,o_].T, t1)
                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,kc,ia,ia", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core_a[v_,o_].T, t1)                # TPD Contributor - Remove
                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,fn,kc", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,fk,nc", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,cn,kf", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,ck,nf", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)                # TPD Contributor - Remove

                        # Doubles/Singles terms.
                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)

                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ia,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core_a[v_,o_].T, 2*t2 - t2.swapaxes(2,3))

                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,ia,ikaf", t1, U_H[beta][v_,v_], U_R[o_,v_] + half_S_core_a[v_,o_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical # TPD Contributor - Remove
                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ia,inac", t1, U_H[beta][o_,o_], U_R[o_,v_] + half_S_core_a[v_,o_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical # TPD Contributor - Remove
                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,ik,incf", t1, U_H[beta][v_,o_], U_R[o_,o_] + half_S_core_a[o_,o_].T, 2*t2 - t2.swapaxes(2,3))             # TPD Contributor - Remove
                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,in,ikfc", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core_a[o,o_].T, 2*t2 - t2.swapaxes(2,3))
                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,ca,knaf", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core_a[v_,v_].T, 2*t2 - t2.swapaxes(2,3))             # TPD Contributor - Remove
                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,fa,knca", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core_a[v_,v_].T, 2*t2 - t2.swapaxes(2,3))

                        # Singles/Doubles terms.
                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,dl,kc", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ea,kice", t1, U_H[beta][o_,v_], U_R[v_,v_] + half_S_core_a[v_,v_].T, 2*t2 - t2.swapaxes(2,3))             # TPD Contributor - Remove
                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,kc,im,kmca", t1, U_H[beta][o_,v_], U_R[o_,o_] + half_S_core_a[o_,o_].T, 2*t2 - t2.swapaxes(2,3))             # TPD Contributor - Remove
                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ac,em,imce", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core_a[o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical # TPD Contributor - Remove
                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ec,em,imac", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core_a[o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical
                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,ki,em,kmae", t1, U_H[beta][o_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical # TPD Contributor - Remove
                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core_a[o,v_].T, 2*t2 - t2.swapaxes(2,3))

                        # Doubles/Doubles terms.
                        AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical

                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)

                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core_a[v_, v_].T) # Canonical
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        # Adding terms for full normalization. 
                        if normalization == 'full':
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ai", t1, U_H[beta][v_, o_])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("kc,kc", t1, U_H[beta][o_, v_])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("kc,cf,kf", t1, U_H[beta][v_,v_], t1) # Canonical
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("kc,nk,nc", t1, U_H[beta][o_,o_], t1) # Canonical
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,kc,ikac", t1, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

                    if orbitals == 'non-canonical':
                        # Computing the Hartree-Fock term of the AAT.
                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        # Singles/Refence terms.
                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_H[beta][v_,o_])

                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core_a[v_,v_].T)
                        AAT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core_a[o,o_].T)

                        # Reference/Singles terms.
                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dT1_dH[beta], U_R[v_,o_] + half_S_core_a[o_,v_].T)

                        AAT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,im,am", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core_a[o,v_].T)

                        # Singles/Singles terms.
                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dH[beta])

                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], U_R[v_,v_] + half_S_core_a[v_,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], U_R[o_,o_] + half_S_core_a[o_,o_].T, t1)

                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,im,km,ka", t1, U_H[beta][o_,o], U_R[o_,o] + half_S_core_a[o,o_].T, t1)
                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,ia,kc,kc", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core_a[v_,o_].T, t1)
                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,em,ia", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,ei,ma", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,am,ie", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,ai,me", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

                        # Doubles/Singles terms.
                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)

                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core_a[v_,o_].T, 2*t2 - t2.swapaxes(2,3))

                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,km,kiea", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core_a[o,o_].T, 2*t2 - t2.swapaxes(2,3))
                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,em,ec,imac", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core_a[v_,v_].T, 2*t2 - t2.swapaxes(2,3))

                        # Singles/Doubles terms.
                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core_a[o,v_].T, 2*t2 - t2.swapaxes(2,3))

                        # Doubles/Doubles terms.
                        AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core_a[o_, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core_a[v_, v_].T)

                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core_a[o, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                        # Adding terms for full normalization. 
                        if normalization == 'full':
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

                            AAT_Norm[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ia,ai", t1, U_H[beta][v_, o_])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)

                    ######### Test #########
                    #if orbitals == 'non-canonical' and self.parameters['freeze_core'] == False:
                    #    # Computing the Hartree-Fock term of the AAT.
                    #    AAT_HF_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                    #    # Singles/Refence terms.
                    #    AAT_S0_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_H[beta][v_,o_])

                    #    AAT_S0_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], -0.5*S_core[a][v_,v_] + half_S_core_a[v_,v_].T)
                    #    AAT_S0_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], -0.5*S_core[a][o_,o] + half_S_core_a[o,o_].T)

                    #    # Reference/Singles terms.
                    #    AAT_0S_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dT1_dH[beta], U_R[v_,o_] + half_S_core_a[o_,v_].T)

                    #    # Singles/Singles terms.
                    #    AAT_SS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dH[beta])

                    #    AAT_SS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], -0.5*S_core[a][v_,v_] + half_S_core_a[v_,v_].T, t1)
                    #    AAT_SS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], -0.5*S_core[a][o_,o_] + half_S_core_a[o_,o_].T, t1)

                    #    AAT_SS_test[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,ia,kc,kc", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core_a[v_,o_].T, t1)
                    #    AAT_SS_test[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,em,ia", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
                    #    AAT_SS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,ei,ma", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)
                    #    AAT_SS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,am,ie", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
                    #    AAT_SS_test[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,ai,me", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

                    #    # Doubles/Singles terms.
                    #    AAT_DS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)

                    #    AAT_DS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core_a[v_,o_].T, 2*t2 - t2.swapaxes(2,3))

                    #    AAT_DS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,km,kiea", t1, U_H[beta][v_,o], -0.5*S_core[a][o_,o] + half_S_core_a[o,o_].T, 2*t2 - t2.swapaxes(2,3))
                    #    AAT_DS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,em,ec,imac", t1, U_H[beta][v_,o_], -0.5*S_core[a][v_,v_] + half_S_core_a[v_,v_].T, 2*t2 - t2.swapaxes(2,3))

                    #    # Singles/Doubles terms.
                    #    AAT_SD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

                    #    AAT_SD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

                    #    # Doubles/Doubles terms.
                    #    AAT_DD_test[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                    #    AAT_DD_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, -0.5*S_core[a][o_, o_] + half_S_core_a[o_, o_].T)
                    #    AAT_DD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, -0.5*S_core[a][v_, v_] + half_S_core_a[v_, v_].T)

                    #    AAT_DD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
                    #    AAT_DD_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
                    #    AAT_DD_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

                    #    # Adding terms for full normalization. 
                    #    if normalization == 'full':
                    #        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

                    #        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ia,ai", t1, U_H[beta][v_, o_]) 
                    #        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
                    #        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)

                    #    assert AAT_HF_test.all() == AAT_HF.all()
                    #    assert AAT_S0_test.all() == AAT_S0.all()
                    #    assert AAT_0S_test.all() == AAT_0S.all()
                    #    assert AAT_SS_test.all() == AAT_SS.all()
                    #    assert AAT_DS_test.all() == AAT_DS.all()
                    #    assert AAT_SD_test.all() == AAT_SD.all()
                    #    assert AAT_DD_test.all() == AAT_DD.all()
                    #    assert AAT_Norm_test.all() == AAT_Norm.all()

                    ########### End ##########

        print("Hartree-Fock AAT:")
        print(AAT_HF, "\n")
        print("Singles/Reference AAT:")
        print(AAT_S0, "\n")
        print("Reference/Singles AAT:")
        print(AAT_0S, "\n")
        print("Singles/Singles AAT:")
        print(AAT_SS, "\n")
        print("Doubles/Singles:")
        print(AAT_DS, "\n")
        print("Singles/Doubles:")
        print(AAT_SD, "\n")
        print("Doubles/Doubles:")
        print(AAT_DD, "\n")

        AAT = AAT_HF + AAT_S0 + AAT_0S + AAT_SS + AAT_DS + AAT_SD + AAT_DD + AAT_Norm

        return AAT



    def compute_CID_AATs(self, normalization='full', orbitals='non-canonical', print_level=0):
        """Compute analytic CID atomic axial tensors.

        Parameters
        ----------
        normalization : {'full', 'intermediate'}, optional
            Wavefunction normalization convention (default ``'full'``).
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).
        print_level : int, optional
            Verbosity level; 0 (default) suppresses output.

        Returns
        -------
        ndarray, shape (3*natom, 3)
            CID AAT ``dI_beta / dR_alpha`` [a.u.].
        """
        # Compute T2 amplitudes and MP2 energy.
        wfn_CID = ci_wfn(self.parameters, self.wfn)
        E_CID, t2 = wfn_CID.solve_CID()

        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3))
        AAT_DD = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        # Compute normalization factor.
        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        # Set up derivative t-amplitude matrices.
        dT2_dH = []

        # Set up U-coefficient matrices for AAT calculations.
        U_H = []

        # Compute OPD and TPD matrices for use in computing the energy gradient.
        # Compute normalize amplitudes.
        N = 1 / np.sqrt(1**2 + oe.contract('ijab,ijab->', np.conjugate(t2), 2*t2-t2.swapaxes(2,3)))
        t0_n = N.copy()
        t2_n = t2 * N

        # Build OPD.
        D_pq = np.zeros_like(F)
        D_pq[o_,o_] -= 2 * oe.contract('jkab,ikab->ij', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[v_,v_] += 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq = D_pq[t_,t_]

        # Build TPD.
        D_pqrs = np.zeros_like(ERI)
        D_pqrs[o_,o_,o_,o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[v_,v_,v_,v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[v_,o_,o_,v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), 2*t2_n - t2_n.swapaxes(2,3))

        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)

        D_pqrs[o_,o_,v_,v_] += np.conjugate(t0_n) * (2*t2_n -t2_n.swapaxes(2,3))
        D_pqrs[v_,v_,o_,o_] += np.conjugate(2*t2_n.swapaxes(0,2).swapaxes(1,3) - t2_n.swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)) * t0_n
        D_pqrs = D_pqrs[t_,t_,t_,t_]

        A_mag, G_mag = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_core = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
            B = h_core[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_h = np.zeros((nbf,nbf))
            U_h[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_h[o,v] += U_h[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_core[o,o].copy() + oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_h[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_core[v,v].copy() + oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                U_h[v,v] += B/D

                for j in range(no):
                    U_h[j,j] = 0
                for c in range(no,nbf):
                    U_h[c,c] = 0

            if orbitals == 'non-canonical':
                U_h[f_,f_] = 0
                U_h[o_,o_] = 0
                U_h[v_,v_] = 0

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            df_dH = np.zeros((nbf,nbf))

            df_dH[o,o] -= h_core[o,o].copy()
            df_dH[o,o] += U_h[o,o] * self.wfn.eps[o].reshape(-1,1) - U_h[o,o].swapaxes(0,1) * self.wfn.eps[o]
            df_dH[o,o] += oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            df_dH[v,v] -= h_core[v,v].copy()
            df_dH[v,v] += U_h[v,v] * self.wfn.eps[v].reshape(-1,1) - U_h[v,v].swapaxes(0,1) * self.wfn.eps[v]
            df_dH[v,v] += oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            dERI_dH =  oe.contract('tr,pqts->pqrs', U_h[:,t], ERI[t,t,:,t])
            dERI_dH += oe.contract('ts,pqrt->pqrs', U_h[:,t], ERI[t,t,t,:])
            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_h[:,t], ERI[:,t,t,t])
            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_h[:,t], ERI[t,:,t,t])

            # Compute CISD energy gradient.
            dE_dH = oe.contract('pq,pq->', df_dH[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dH[t_,t_,t_,t_], D_pqrs)

            # Computing the HF energy gradient.
            dE_dH_HF = 2 * oe.contract('ii->', h_core[o,o])
            dE_dH_tot = dE_dH + dE_dH_HF

            # Compute dT2_dR guess amplitudes.
            dt2_dH = -dE_dH * t2
            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
            dt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
            dt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
            dt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH /= wfn_CID.D_ijab

            # Solve for initial CISD energy gradient.
            dE_dH_proj =  oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
            dE_dH_proj += oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
            dt2_dH = dt2_dH.copy()

            # Start iterative procedure.
            iteration = 1
            while iteration <= self.parameters['max_iterations']:
                dE_dH_proj_old = dE_dH_proj
                dt2_dH_old = dt2_dH.copy()

                # Solving for the derivative residuals.
                dRt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                dRt2_dH -= dE_dH_proj * t2
                dRt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
                dRt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
                dRt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
                dRt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
                dRt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
                dRt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
                dRt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
                dRt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)

                dRt2_dH -= E_CID * dt2_dH
                dRt2_dH += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dH)
                dRt2_dH += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dH)
                dRt2_dH -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dH)
                dRt2_dH += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dH)
                dRt2_dH -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dH)
                dRt2_dH -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dH)

                dt2_dH += dRt2_dH / wfn_CID.D_ijab

                # Perform DIIS extrapolation.
                if self.parameters['DIIS']:
                    occ = len(dt2_dH)
                    vir = len(dt2_dH[0][0])
                    dt2_dH_flat = len(np.reshape(dt2_dH, (-1)))
                    res_vec = np.reshape(dRt2_dH, (-1))
                    t_vec = np.reshape(dt2_dH, (-1))
                    if iteration == 1:
                        t_iter = np.atleast_2d(t_vec).T
                        e_iter = np.atleast_2d(res_vec).T
                    t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                    dt2_dH = np.reshape(t_vec, (occ, occ, vir, vir))

                # Compute new CISD energy gradient.
                dE_dH_proj =  oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dH_proj += oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                # Compute new total energy gradient.
                dE_dH_tot_proj = dE_dH_proj + dE_dH_HF

                # Compute convergence data.
                rms_dt2_dH = oe.contract('ijab,ijab->', dt2_dH_old - dt2_dH, dt2_dH_old - dt2_dH)
                rms_dt2_dH = np.sqrt(rms_dt2_dH)
                delta_dE_dH_proj = dE_dH_proj_old - dE_dH_proj

                if print_level > 0:
                    print(" %02d %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dH_proj, dE_dH_tot_proj, delta_dE_dH_proj, rms_dt2_dH))

                if iteration > 1:
                    if abs(delta_dE_dH_proj) < self.parameters['e_convergence'] and rms_dt2_dH < self.parameters['d_convergence']:
                        #print("Convergence criteria met.")
                        break
                if iteration == self.parameters['max_iterations']:
                    if abs(delta_dE_dH_proj) > self.parameters['e_convergence'] or rms_dt2_dH > self.parameters['d_convergence']:
                        print("Not converged.")
                iteration += 1

            print("\nMagnetic Field Perturbation Data:")
            print("Cartesian: ", a)
            print("Maximum dt2/dH: ", np.max(dt2_dH))

            dT2_dH.append(dt2_dH)
            U_H.append(U_h)

        # Delete excess variables.
        #del dERI_dH; del dt1_dH; del dt2_dH; del dRt1_dH; del dRt2_dH; del dt1_dH_old; del dt2_dH_old
        #del df_dH; del h_core; del B; del U_h; del A_mag; del G_mag
        #gc.collect()

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            # Compute the half derivative overlap for AAT calculation.
            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_core[a] = T_core[a].np
                V_core[a] = V_core[a].np
                S_core[a] = S_core[a].np

                ERI_core[a] = ERI_core[a].np
                ERI_core[a] = ERI_core[a].swapaxes(1,2)
                half_S_core[a] = half_S_core[a].np

                # Computing skeleton (core) first derivative integrals.
                h_core = T_core[a] + V_core[a]
                F_core = T_core[a] + V_core[a] + oe.contract('piqi->pq', 2 * ERI_core[a][:,o,:,o] - ERI_core[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_core[v,o] + oe.contract('ai,ii->ai', S_core[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_core[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_R = np.zeros((nbf,nbf))
                U_R[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_R[o,v] -= U_R[v,o].T + S_core[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_core[o,o].copy() - oe.contract('ij,jj->ij', S_core[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_R[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_core[v,v].copy() - oe.contract('ab,bb->ab', S_core[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_R[v,v] += B/D

                    for j in range(no):
                        U_R[j,j] = -0.5 * S_core[a][j,j]
                    for c in range(no,nbf):
                        U_R[c,c] = -0.5 * S_core[a][c,c]

                if orbitals == 'non-canonical':
                    U_R[f_,f_] = -0.5 * S_core[a][f_,f_]
                    U_R[o_,o_] = -0.5 * S_core[a][o_,o_]
                    U_R[v_,v_] = -0.5 * S_core[a][v_,v_]

                # Computing the gradient of the Fock matrix.
                df_dR = np.zeros((nbf,nbf))

                df_dR[o,o] += F_core[o,o].copy()
                df_dR[o,o] += U_R[o,o] * self.wfn.eps[o].reshape(-1,1) + U_R[o,o].swapaxes(0,1) * self.wfn.eps[o]
                df_dR[o,o] += oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o])
                df_dR[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                df_dR[v,v] += F_core[v,v].copy()
                df_dR[v,v] += U_R[v,v] * self.wfn.eps[v].reshape(-1,1) + U_R[v,v].swapaxes(0,1) * self.wfn.eps[v]
                df_dR[v,v] += oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o])
                df_dR[v,v] -= 0.5 * oe.contract('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])

                # Computing the gradient of the ERIs.
                dERI_dR = ERI_core[a].copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:,t], ERI[:,t,t,t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:,t], ERI[t,:,t,t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:,t], ERI[t,t,:,t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:,t], ERI[t,t,t,:])

                # Compute CISD energy gradient.
                dE_dR = oe.contract('pq,pq->', df_dR[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dR[t_,t_,t_,t_], D_pqrs)

                # Computing the HF energy gradient.
                dE_dR_HF = 2 * oe.contract('ii->', h_core[o,o])
                dE_dR_HF += oe.contract('ijij->', 2 * ERI_core[a][o,o,o,o] - ERI_core[a].swapaxes(2,3)[o,o,o,o])
                dE_dR_HF -= 2 * oe.contract('ii,i->', S_core[a][o,o], self.wfn.eps[o])
                dE_dR_HF += Nuc_Gradient[N1][a]

                dE_dR_tot = dE_dR + dE_dR_HF

                # Compute dT2_dR guess amplitudes.
                dt2_dR = -dE_dR * t2
                dt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2)
                dt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2)
                dt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2)
                dt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2)
                dt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2)
                dt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)
                dt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2)
                dt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                dt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR /= wfn_CID.D_ijab

                # Solve for initial CISD energy gradient.
                dE_dR_proj =  oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dR_proj += oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
                dt2_dR = dt2_dR.copy()

                # Start iterative procedure.
                iteration = 1
                while iteration <= self.parameters['max_iterations']:
                    dE_dR_proj_old = dE_dR_proj
                    dt2_dR_old = dt2_dR.copy()

                    # Solving for the derivative residuals.
                    dRt2_dR = dERI_dR.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                    dRt2_dR -= dE_dR_proj * t2
                    dRt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2)
                    dRt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)

                    dRt2_dR -= E_CID * dt2_dR
                    dRt2_dR += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dR)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dR)

                    dt2_dR += dRt2_dR / wfn_CID.D_ijab

                    # Perform DIIS extrapolation.
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

                    # Compute new CISD energy gradient.
                    dE_dR_proj =  oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                    dE_dR_proj += oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                    # Compute new total energy gradient.
                    dE_dR_tot_proj = dE_dR_proj + dE_dR_HF

                    # Compute convergence data.
                    rms_dt2_dR = oe.contract('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR)
                    rms_dt2_dR = np.sqrt(rms_dt2_dR)
                    delta_dE_dR_proj = dE_dR_proj_old - dE_dR_proj

                    if print_level > 0:
                        print(" %02d %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dR_proj, dE_dR_tot_proj, delta_dE_dR_proj, rms_dt2_dR))

                    if iteration > 1:
                        if abs(delta_dE_dR_proj) < self.parameters['e_convergence'] and rms_dt2_dR < self.parameters['d_convergence']:
                            #print("Convergence criteria met.")
                            break
                    if iteration == self.parameters['max_iterations']:
                        if abs(delta_dE_dR_proj) > self.parameters['e_convergence'] or rms_dt2_dR > self.parameters['d_convergence']:
                            print("Not converged.")
                    iteration += 1

                print("\nNuclear Perturbation Data:")
                print("Atom: ", N1)
                print("Cartesian: ", a)
                print("Maximum dt2/dR: ", np.max(dt2_dR))

                # Compute derivative of the normalization factor.
                N_R = - (1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_R *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))

                for beta in range(0,3):
                    #Setting up AAT indexing.
                    lambda_alpha = 3 * N1 + a

                    if orbitals == 'canonical':
                        # Computing the Hartree-Fock term of the AAT.
                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        # Doubles/Doubles terms.
                        AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical

                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)

                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core[a][v_, v_].T) # Canonical
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        # Adding terms for full normalization. 
                        if normalization == 'full':
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

                    if orbitals == 'non-canonical':
                        # Computing the Hartree-Fock term of the AAT.
                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        # Doubles/Doubles terms.
                        AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)

                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                        # Adding terms for full normalization. 
                        if normalization == 'full':
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

        print("Hartree-Fock AAT:")
        print(AAT_HF, "\n")
        print("Doubles/Doubles:")
        print(AAT_DD, "\n")

        AAT = AAT_HF + AAT_DD + AAT_Norm

        return AAT



    def compute_RHF_DO_AATs(self, orbitals='non-canonical'):
        """Compute analytic RHF AATs using the distributed-origin gauge.

        Parameters
        ----------
        orbitals : {'non-canonical', 'canonical'}, optional
            Orbital convention (default ``'non-canonical'``).

        Returns
        -------
        ndarray, shape (3*natom, 3)
            Distributed-origin RHF AAT ``dI_beta / dR_alpha`` [a.u.].
        """
        m = self._setup_mo_basis()
        C, nbf, no, nv = m.C, m.nbf, m.no, m.nv
        f_, o_, v_, t_ = m.f_, m.o_, m.v_, m.t_
        o, v, t = m.o, m.v, m.t
        C_p4, natom, atoms = m.C_p4, m.natom, m.atoms
        h, ERI, F, mints = m.h, m.ERI, m.F, m.mints
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Set up U-coefficient matrices for AAT calculations.
        U_R = [] 
        U_H = [] 

        A, G = self._build_cphf_A(ERI, F, no, nv, o, v, sign=+1)

        # First derivative matrices.
        half_S = [] 

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_d1 = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_d1 = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_d1 = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_d1 = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            # Compute the half derivative overlap for AAT calculation.
            half_S_d1 = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_d1[a] = T_d1[a].np
                V_d1[a] = V_d1[a].np
                S_d1[a] = S_d1[a].np

                ERI_d1[a] = ERI_d1[a].np
                ERI_d1[a] = ERI_d1[a].swapaxes(1,2)
                half_S_d1[a] = half_S_d1[a].np

                # Computing skeleton (core) first derivative integrals.
                h_d1 = T_d1[a] + V_d1[a]
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + oe.contract('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf))
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_d1[o,o].copy() - oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_d1[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_d1[v,v].copy() - oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_d1[v,v] += B/D

                    for j in range(no):
                        U_d1[j,j] = -0.5 * S_d1[a][j,j]
                    for b in range(no,nbf):
                        U_d1[b,b] = -0.5 * S_d1[a][b,b]

                if orbitals == 'non-canonical':
                    U_d1[f_,f_] = -0.5 * S_d1[a][f_,f_]
                    U_d1[o_,o_] = -0.5 * S_d1[a][o_,o_]
                    U_d1[v_,v_] = -0.5 * S_d1[a][v_,v_]

                # Appending to lists.
                half_S.append(half_S_d1[a])
                U_R.append(U_d1)

        ###
        A_mag, G_mag = self._build_cphf_A(ERI, F, no, nv, o, v, sign=-1)

        # Get the angular momentum and nabla AO integrals.
        L_psi4 = [mints.ao_angular_momentum()[i].np for i in range(3)]
        nabla_AO = [mints.ao_nabla()[i].np for i in range(3)]

        # Levi-Civita tensor.
        eps_lc = np.zeros((3, 3, 3))
        eps_lc[0,1,2] = eps_lc[1,2,0] = eps_lc[2,0,1] = 1
        eps_lc[0,2,1] = eps_lc[2,1,0] = eps_lc[1,0,2] = -1

        # Get geometry for origin shifting.
        geom_bohr = np.array(self.H.molecule.geometry().np)

        # For each atom, build atom-centered angular momentum integrals
        # and solve the CPHF for the magnetic field perturbation.
        # U_H[A][beta] = CPHF solution for atom A, magnetic field component beta.
        U_H = []
        for A in range(natom):
            U_H_A = []
            for beta in range(3):
                # Shift angular momentum origin to atom A:
                # L(R_A) = L(0) - R_A x nabla
                L_shifted = L_psi4[beta].copy()
                for gamma in range(3):
                    for delta in range(3):
                        #L_shifted -= eps_lc[beta, gamma, delta] * geom_bohr[A, gamma] * nabla_AO[delta]
                        L_shifted += eps_lc[beta, gamma, delta] * geom_bohr[A, gamma] * nabla_AO[delta]

                # Build atom-centered magnetic dipole integrals in MO basis.
                mu_mag_shifted = -0.5 * L_shifted
                mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_shifted, C)

                # Compute the perturbation-dependent B matrix.
                B = mu_mag[v, o]

                # Solve the CPHF.
                U_d1 = np.zeros((nbf, nbf))
                U_d1[v, o] += (G_mag @ B.reshape((nv * no))).reshape(nv, no)
                U_d1[o, v] += U_d1[v, o].T

                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B_oo = -mu_mag[o,o].copy() + oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                    U_d1[o,o] += B_oo / D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B_vv = -mu_mag[v,v].copy() + oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                    U_d1[v,v] += B_vv / D

                    for j in range(no):
                        U_d1[j, j] = 0
                    for b in range(no, nbf):
                        U_d1[b, b] = 0

                if orbitals == 'non-canonical':
                    U_d1[f_, f_] = 0
                    U_d1[o_, o_] = 0
                    U_d1[v_, v_] = 0

                U_H_A.append(U_d1)
            U_H.append(U_H_A)

        # Setting up the AATs.
        AAT_HF = np.zeros((natom * 3, 3))

        # Compute AATs using atom-centered magnetic field solutions.
        for lambda_alpha in range(3 * natom):
            A = lambda_alpha // 3  # atom index for this row
            for beta in range(3):
                # Use U_H[A][beta] — the CPHF solution with origin at atom A.
                AAT_HF[lambda_alpha][beta] += 2 * oe.contract("em,em", U_H[A][beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

        AAT = AAT_HF

        return AAT








