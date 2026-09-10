"""Contains the coupled cluster (CC) wavefunction object."""

import warnings
import psi4
import numpy as np
import scipy.linalg as la
import opt_einsum as oe
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.utils import get_slices
from apyib.utils import compute_F_MO
from apyib.utils import compute_ERI_MO
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO
from apyib.utils import solve_general_DIIS

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)
class cc_wfn(object):
    """
    Wavefunction object.
    """
    # Define the specific properties of the CC wavefunction.
    def __init__(self, parameters, wfn):
        """Initialise the CC wavefunction from a converged RHF wavefunction.

        Parameters
        ----------
        parameters : dict
            Calculation parameters.
        wfn : hf_wfn
            Converged RHF wavefunction.
        """
        # Define the Hamiltonian and the Hartree-Fock reference energy and wavefunction.
        self.parameters = parameters
        self.H = wfn.H
        self.wfn = wfn
        self.C = wfn.C

        # Get slice lists for frozen core, occupied, virtual, and total orbital subspaces.
        self.C_list, self.I_list = get_slices(self.parameters, self.wfn)

        # Setting up slice options for energy denominators.
        o = self.C_list[1]
        v = self.C_list[2]

        # Build energy denominators.
        self.eps_o = wfn.eps[o]
        self.eps_v = wfn.eps[v]
        self.D_ia = self.eps_o.reshape(-1,1) - self.eps_v
        self.D_ijab = self.eps_o.reshape(-1,1,1,1) + self.eps_o.reshape(-1,1,1) - self.eps_v.reshape(-1,1) - self.eps_v

        # Set up MO one- and two-electron integrals.
        self.F_MO, E_fc = compute_F_MO(self.parameters, self.wfn, self.C_list)
        self.ERI_MO = compute_ERI_MO(self.parameters, self.wfn, self.C_list)



    def solve_CCSD_SO(self, print_level=0, solving_lambda=False):
        """
        Solves the CCSD amplitudes and energies in the spin-orbital basis.

        The amplitude equations are the projections of the similarity-transformed
        Hamiltonian onto the singles and doubles manifolds,

            0 = < Phi_i^a  | H_bar | Phi_0 >
            0 = < Phi_ij^ab | H_bar | Phi_0 >

        with the correlation energy evaluated separately as < Phi_0 | H_bar | Phi_0 >.
        These equations are projective, not an eigenvalue problem, so the residuals
        carry no ``-E * t`` term.

        DIIS extrapolation is applied to the amplitude increments when
        ``parameters['DIIS']`` is set.

        Parameters
        ----------
        print_level : int, optional
            Verbosity level. ``> 0`` prints the iteration table.
        solving_lambda : bool, optional
            Whether to additionally solve the Lambda (de-excitation) amplitude
            equations, using the converged t amplitudes.

        Returns
        -------
        E_CCSD : float or complex
            CCSD correlation energy.
        t1 : ndarray
            Converged singles amplitudes, shape ``(no, nv)``.
        t2 : ndarray
            Converged doubles amplitudes, shape ``(no, no, nv, nv)``.
        l1 : ndarray or int
            Converged Lambda singles amplitudes, or ``0`` when
            ``solving_lambda`` is False.
        l2 : ndarray or int
            Converged Lambda doubles amplitudes, or ``0`` when
            ``solving_lambda`` is False.

        Notes
        -----
        The return length is fixed at five regardless of ``solving_lambda``;
        unsolved Lambda amplitudes come back as ``0``, matching the sentinel
        convention already used for the singles of methods that lack them.
        """
        # Sentinels for the case where the Lambda equations are not requested.
        l1 = 0
        l2 = 0

        # Build energy denominators in the spin orbital basis.
        eps_o = np.repeat(self.eps_o, 2)
        eps_v = np.repeat(self.eps_v, 2)
        D_ia = eps_o.reshape(-1,1) - eps_v
        D_ijab = eps_o.reshape(-1,1,1,1) + eps_o.reshape(-1,1,1) - eps_v.reshape(-1,1) - eps_v

        # Setting up slice options for the MO integrals.
        o_ = self.I_list[1]
        v_ = self.I_list[2]

        # Compute F_MO in spin-orbital basis.
        F_SO = compute_F_SO(self.wfn, self.F_MO)

        # Compute ERI_MO in spin-orbital basis.
        ERI_SO = compute_ERI_SO(self.wfn, self.ERI_MO)

        # Swap axes for Dirac notation.
        ERI_SO = ERI_SO.swapaxes(1,2)       # (pr|qs) -> <pq|rs>

        # Antisymmetrized two-electron integrals, <pq||rs> = <pq|rs> - <pq|sr>.
        # The CCSD residuals reference these in nearly every term, so they are
        # formed once here rather than inline as in the CI solvers.
        G = ERI_SO - ERI_SO.swapaxes(2,3)

        # Initial T1 guess amplitude.
        t1 = F_SO.copy().swapaxes(0,1)[o_,v_] / D_ia

        # Initial T2 guess amplitude.
        t2 = G.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] / D_ijab

        # Solve for initial CCSD energy.
        E_CCSD = self._energy_SO(F_SO, G, t1, t2, o_, v_)
        t1 = t1.copy()
        t2 = t2.copy()

        if print_level > 0:
            print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_T1(real)      RMS_T1(imaginary)      RMS_T2(real)      RMS_T2(imaginary)")

        # Start iterative procedure.
        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_CCSD_old = E_CCSD
            t1_old = t1.copy()
            t2_old = t2.copy()

            # Solving for the residuals. The T2 equations are all permuted to have
            # final indices i,j,a,b for T_ijab.
            r_T1, r_T2 = self._residuals_SO(F_SO, G, t1, t2, o_, v_)

            # Amplitude increments, t_new - t_old. These are the DIIS error
            # vectors: the extrapolation acts on the change in the amplitudes
            # rather than on the bare residual.
            dt1 = r_T1 / D_ia
            dt2 = r_T2 / D_ijab

            t1 += dt1
            t2 += dt2

            # Perform DIIS extrapolation.
            if self.parameters['DIIS']:
                occ = len(t1)
                vir = len(t1[0])
                t1_flat = len(np.reshape(t1, (-1)))
                t2_flat = len(np.reshape(t2, (-1)))
                res_vec = np.concatenate((np.reshape(dt1, (-1)), np.reshape(dt2, (-1))))
                t_vec = np.concatenate((np.reshape(t1, (-1)), np.reshape(t2, (-1))))
                if iteration == 1:
                    t_iter = np.atleast_2d(t_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                t1 = np.reshape(t_vec[0:t1_flat], (occ, vir))
                t2 = np.reshape(t_vec[t1_flat:], (occ, occ, vir, vir))

            # Compute new CCSD energy.
            E_CCSD = self._energy_SO(F_SO, G, t1, t2, o_, v_)

            # Compute total energy.
            E_tot = self.wfn.E_SCF + E_CCSD + self.H.E_nuc

            # Compute convergence data.
            rms_t1 = oe.contract('ia,ia->', t1_old - t1, t1_old - t1)
            rms_t1 = np.sqrt(rms_t1)

            rms_t2 = oe.contract('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CCSD_old - E_CCSD

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CCSD.real, E_CCSD.imag, E_tot, delta_E.real, delta_E.imag, rms_t1.real, rms_t1.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t1 < self.parameters['d_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    warnings.warn(
                        f"CCSD (SO) did not converge in {self.parameters['max_iterations']} iterations. "
                        f"|ΔE| = {abs(delta_E):.2e}, RMS(T2) = {float(rms_t2.real):.2e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            iteration += 1

        # The Lambda equations take the converged t amplitudes as fixed input, so
        # they are solved once here rather than alongside the T iteration.
        if solving_lambda:
            l1, l2 = self._solve_lambda_SO(F_SO, G, t1, t2, D_ia, D_ijab, o_, v_,
                                           print_level)

        return E_CCSD, t1, t2, l1, l2



    def _energy_SO(self, F, G, t1, t2, o_, v_):
        """
        Computes the CCSD correlation energy, < Phi_0 | H_bar | Phi_0 >.

            E = f_kc t_k^c + 1/4 <kl||cd> t_kl^cd + 1/2 <kl||cd> t_k^c t_l^d
        """
        E  = oe.contract('kc,kc->', F[o_,v_], t1)
        E += 0.25 * oe.contract('klcd,klcd->', G[o_,o_,v_,v_], t2)
        E += 0.5 * oe.contract('klcd,kc,ld->', G[o_,o_,v_,v_], t1, t1)

        return E



    def _residuals_SO(self, F, G, t1, t2, o_, v_):
        """
        Computes the CCSD singles and doubles residuals.

        Builds the effective amplitudes and the intermediates shared by the two
        residuals, then evaluates both.

        Returns
        -------
        r_T1 : ndarray
            Singles residual, shape ``(no, nv)``.
        r_T2 : ndarray
            Doubles residual, shape ``(no, no, nv, nv)``.
        """
        tau, tau_tilde = self._tau_s_SO(t1, t2)

        F_ae = self._F_ae_SO(F, G, t1, tau_tilde, o_, v_)
        F_mi = self._F_mi_SO(F, G, t1, tau_tilde, o_, v_)
        F_me = self._F_me_SO(F, G, t1, o_, v_)

        W_mnij = self._W_mnij_SO(G, t1, tau, o_, v_)
        W_abef = self._W_abef_SO(G, t1, o_, v_)
        W_mbej = self._W_mbej_SO(G, t1, t2, o_, v_)

        r_T1 = self._singles_residual_SO(F, G, t1, t2, F_ae, F_mi, F_me, o_, v_)
        r_T2 = self._doubles_residual_SO(G, t1, t2, tau, F_ae, F_mi, F_me,
                                         W_mnij, W_abef, W_mbej, o_, v_)

        return r_T1, r_T2



    def _singles_residual_SO(self, F, G, t1, t2, F_ae, F_mi, F_me, o_, v_):
        """
        Computes the CCSD singles residual, < Phi_i^a | H_bar | Phi_0 >.

            r_i^a = f_ai + t_i^e F_ae - t_m^a F_mi + t_im^ae F_me
                         + <ma||ei> t_m^e + 1/2 <ma||ef> t_mi^ef
                         - 1/2 <mn||ei> t_mn^ea
        """
        # f_ai, not f_ia; the two differ for complex orbitals.
        r_T1  = F.copy().swapaxes(0,1)[o_,v_]

        # Terms carried by the one-particle intermediates.
        r_T1 += oe.contract('ie,ae->ia', t1, F_ae)
        r_T1 -= oe.contract('ma,mi->ia', t1, F_mi)
        r_T1 += oe.contract('imae,me->ia', t2, F_me)

        # Terms with no intermediate to absorb them.
        r_T1 += oe.contract('maei,me->ia', G[o_,v_,v_,o_], t1)
        r_T1 += 0.5 * oe.contract('maef,mief->ia', G[o_,v_,v_,v_], t2)
        r_T1 -= 0.5 * oe.contract('mnei,mnea->ia', G[o_,o_,v_,o_], t2)

        return r_T1



    def _doubles_residual_SO(self, G, t1, t2, tau, F_ae, F_mi, F_me,
                               W_mnij, W_abef, W_mbej, o_, v_):
        """
        Computes the CCSD doubles residual, < Phi_ij^ab | H_bar | Phi_0 >.

            r_ij^ab = <ab||ij>
                    + 1/2 tau_mn^ab W_mnij + 1/2 tau_ij^ef W_abef
                    + P(a,b) t_ij^ae (F_be - 1/2 t_m^b F_me)
                    - P(i,j) t_im^ab (F_mj + 1/2 t_j^e F_me)
                    + P(i,j) P(a,b) (t_im^ae W_mbej - t_i^e t_m^a <mb||ej>)
                    + P(i,j) t_i^e <ab||ej> - P(a,b) t_m^a <mb||ij>

        where P(i,j) X_ijab = X_ijab - X_jiab and P(a,b) X_ijab = X_ijab - X_ijba.
        """
        # Terms already antisymmetric in both index pairs.
        # <ab||ij>, not <ij||ab>; the two differ for complex orbitals.
        r_T2  = G.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
        r_T2 += 0.5 * oe.contract('mnab,mnij->ijab', tau, W_mnij)
        r_T2 += 0.5 * oe.contract('ijef,abef->ijab', tau, W_abef)

        # F_ae and F_mi dressed with the singles, as only the doubles equation needs.
        F_ae_dressed = F_ae - 0.5 * oe.contract('mb,me->be', t1, F_me)
        F_mi_dressed = F_mi + 0.5 * oe.contract('je,me->mj', t1, F_me)

        # Terms carrying P(a,b).
        r_P_ab  = oe.contract('ijae,be->ijab', t2, F_ae_dressed)
        r_P_ab -= oe.contract('ma,mbij->ijab', t1, G[o_,v_,o_,o_])

        # Terms carrying P(i,j).
        r_P_ij  = -oe.contract('imab,mj->ijab', t2, F_mi_dressed)
        r_P_ij += oe.contract('ie,abej->ijab', t1, G[v_,v_,v_,o_])

        # Terms carrying P(i,j) P(a,b).
        r_P_ij_ab  = oe.contract('imae,mbej->ijab', t2, W_mbej)
        r_P_ij_ab -= oe.contract('ie,ma,mbej->ijab', t1, t1, G[o_,v_,v_,o_])

        # Apply each permutation operator once to the group that carries it.
        r_T2 += r_P_ab - r_P_ab.swapaxes(2,3)
        r_T2 += r_P_ij - r_P_ij.swapaxes(0,1)
        r_T2 += (r_P_ij_ab
                 - r_P_ij_ab.swapaxes(0,1)
                 - r_P_ij_ab.swapaxes(2,3)
                 + r_P_ij_ab.swapaxes(0,1).swapaxes(2,3))

        return r_T2



    def _tau_s_SO(self, t1, t2):
        """
        Computes the effective doubles amplitudes tau and tau-tilde.

            tau_ij^ab       = t_ij^ab +       (t_i^a t_j^b - t_i^b t_j^a)
            tau-tilde_ij^ab = t_ij^ab + 1/2 * (t_i^a t_j^b - t_i^b t_j^a)
        """
        t1t1 = oe.contract('ia,jb->ijab', t1, t1)
        t1t1 = t1t1 - t1t1.swapaxes(2,3)

        tau_tilde = t2 + 0.5 * t1t1
        tau = t2 + t1t1

        return tau, tau_tilde



    def _F_ae_SO(self, F, G, t1, tau_tilde, o_, v_):
        """
        Computes the one-particle intermediate F_ae.

            F_ae = f_ae - 1/2 f_me t_m^a + <ma||fe> t_m^f
                        - 1/2 <mn||fe> tau-tilde_mn^fa
        """
        F_ae  = F[v_,v_].copy()
        F_ae -= 0.5 * oe.contract('me,ma->ae', F[o_,v_], t1)
        F_ae += oe.contract('mafe,mf->ae', G[o_,v_,v_,v_], t1)
        F_ae -= 0.5 * oe.contract('mnfe,mnfa->ae', G[o_,o_,v_,v_], tau_tilde)

        return F_ae



    def _F_mi_SO(self, F, G, t1, tau_tilde, o_, v_):
        """
        Computes the one-particle intermediate F_mi.

            F_mi = f_mi + 1/2 f_me t_i^e + <mn||ie> t_n^e
                        - 1/2 <mn||ef> tau-tilde_ni^ef
        """
        F_mi  = F[o_,o_].copy()
        F_mi += 0.5 * oe.contract('me,ie->mi', F[o_,v_], t1)
        F_mi += oe.contract('mnie,ne->mi', G[o_,o_,o_,v_], t1)
        F_mi -= 0.5 * oe.contract('mnef,nief->mi', G[o_,o_,v_,v_], tau_tilde)

        return F_mi



    def _F_me_SO(self, F, G, t1, o_, v_):
        """
        Computes the one-particle intermediate F_me.

            F_me = f_me + <mn||ef> t_n^f
        """
        F_me  = F[o_,v_].copy()
        F_me += oe.contract('mnef,nf->me', G[o_,o_,v_,v_], t1)

        return F_me



    def _W_mnij_SO(self, G, t1, tau, o_, v_):
        """
        Computes the two-particle intermediate W_mnij.

            W_mnij = <mn||ij> + P(i,j) t_j^e <mn||ie> + 1/2 tau_ij^ef <mn||ef>

        where P(i,j) X_mnij = X_mnij - X_mnji.
        """
        W_mnij = G[o_,o_,o_,o_].copy()
        P_ij = oe.contract('mnie,je->mnij', G[o_,o_,o_,v_], t1)
        W_mnij += P_ij - P_ij.swapaxes(2,3)
        W_mnij += 0.5 * oe.contract('ijef,mnef->mnij', tau, G[o_,o_,v_,v_])

        return W_mnij



    def _W_abef_SO(self, G, t1, o_, v_):
        """
        Computes the two-particle intermediate W_abef.

            W_abef = <ab||ef> - P(a,b) t_m^b <am||ef>

        where P(a,b) X_abef = X_abef - X_baef.
        """
        W_abef = G[v_,v_,v_,v_].copy()
        P_ab = oe.contract('mb,amef->abef', t1, G[v_,o_,v_,v_])
        W_abef -= P_ab - P_ab.swapaxes(0,1)

        return W_abef



    def _W_mbej_SO(self, G, t1, t2, o_, v_):
        """
        Computes the two-particle intermediate W_mbej.

            W_mbej = <mb||ej> + t_j^f <mb||ef> - t_n^b <mn||ej>
                              - (1/2 t_jn^fb + t_j^f t_n^b) <mn||ef>
        """
        W_mbej  = G[o_,v_,v_,o_].copy()
        W_mbej += oe.contract('jf,mbef->mbej', t1, G[o_,v_,v_,v_])
        W_mbej -= oe.contract('nb,mnej->mbej', t1, G[o_,o_,v_,o_])
        W_mbej -= oe.contract('jnfb,mnef->mbej',
                              0.5 * t2 + oe.contract('jf,nb->jnfb', t1, t1),
                              G[o_,o_,v_,v_])

        return W_mbej



    def _pseudoenergy_SO(self, F, G, l1, l2, o_, v_):
        """
        Computes the Lambda pseudoenergy, used only to monitor convergence.

            E_pseudo = f_kc l_k^c + 1/4 <kl||cd> l_kl^cd

        This is the CCSD energy functional with the Lambda amplitudes in place of
        the t amplitudes. It is not a physical energy; it is a linear functional of
        Lambda that settles when the Lambda equations are satisfied.
        """
        E  = oe.contract('kc,kc->', F[o_,v_], l1)
        E += 0.25 * oe.contract('klcd,klcd->', G[o_,o_,v_,v_], l2)

        return E



    def _solve_lambda_SO(self, F, G, t1, t2, D_ia, D_ijab, o_, v_, print_level=0):
        """
        Solves the CCSD Lambda amplitudes for a converged set of t amplitudes.

            0 = dL/dt_k^c        0 = dL/dt_kl^cd

        The t amplitudes are held fixed, so these equations are linear in Lambda
        and the iteration is a fixed-point solve rather than a nonlinear one.

        Returns
        -------
        l1 : ndarray
            Converged Lambda singles amplitudes, shape ``(no, nv)``.
        l2 : ndarray
            Converged Lambda doubles amplitudes, shape ``(no, no, nv, nv)``.

        Notes
        -----
        The residuals carry the diagonal Fock terms, which contribute
        ``-D * Lambda``, so the update ``l += r / D`` lands on the solution of the
        equations exactly as it does for the t amplitudes. The denominators are the
        same arrays used by the T iteration.
        """
        # Lambda and t agree at lowest order, which makes t the natural guess.
        l1 = t1.copy()
        l2 = t2.copy()

        # The t amplitudes are fixed here, so every intermediate except the two G
        # terms is an invariant of the iteration and is built once. F_ia is the
        # T-side F_me read at (i,a); it is not a separate object.
        F_ia = self._F_me_SO(F, G, t1, o_, v_)
        F_ea = self._lambda_F_ea_SO(F, G, t1, t2, o_, v_)
        F_im = self._lambda_F_im_SO(F, G, t1, t2, o_, v_)
        W_eima = self._lambda_W_eima_SO(G, t1, t2, o_, v_)
        W_efma = self._lambda_W_efma_SO(F, G, t1, t2, o_, v_)
        W_eimn = self._lambda_W_eimn_SO(F, G, t1, t2, o_, v_)
        W_ijmn = self._lambda_W_ijmn_SO(G, t1, t2, o_, v_)
        W_efab = self._lambda_W_efab_SO(G, t1, t2, o_, v_)

        E_pseudo = self._pseudoenergy_SO(F, G, l1, l2, o_, v_)

        if print_level > 0:
            print("\n Iter    E_pseudo(real)     E_pseudo(imaginary)    Delta_E(real)      Delta_E(imaginary)     RMS_L1(real)       RMS_L1(imaginary)      RMS_L2(real)      RMS_L2(imaginary)")

        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_pseudo_old = E_pseudo
            l1_old = l1.copy()
            l2_old = l2.copy()

            # G depends on Lambda and is the only intermediate rebuilt each cycle.
            G_om = self._lambda_G_om_SO(t2, l2)
            G_ge = self._lambda_G_ge_SO(t2, l2)

            r_L1 = self._lambda_singles_residual_SO(G, t1, l1, l2, F_ia, F_ea, F_im,
                                                    W_eima, W_efma, W_eimn,
                                                    G_om, G_ge, o_, v_)
            r_L2 = self._lambda_doubles_residual_SO(G, t1, l1, l2, F_ia, F_ea, F_im,
                                                    W_eima, W_ijmn, W_efab,
                                                    G_om, G_ge, o_, v_)

            # Amplitude increments, l_new - l_old, which are also the DIIS error
            # vectors, matching the convention used for the t amplitudes.
            dl1 = r_L1 / D_ia
            dl2 = r_L2 / D_ijab

            l1 += dl1
            l2 += dl2

            # Perform DIIS extrapolation.
            if self.parameters['DIIS']:
                occ = len(l1)
                vir = len(l1[0])
                l1_flat = len(np.reshape(l1, (-1)))
                res_vec = np.concatenate((np.reshape(dl1, (-1)), np.reshape(dl2, (-1))))
                l_vec = np.concatenate((np.reshape(l1, (-1)), np.reshape(l2, (-1))))
                if iteration == 1:
                    l_iter = np.atleast_2d(l_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                l_vec, e_iter, l_iter = solve_general_DIIS(self.parameters, res_vec, l_vec, e_iter, l_iter, iteration)
                l1 = np.reshape(l_vec[0:l1_flat], (occ, vir))
                l2 = np.reshape(l_vec[l1_flat:], (occ, occ, vir, vir))

            E_pseudo = self._pseudoenergy_SO(F, G, l1, l2, o_, v_)

            # Compute convergence data.
            rms_l1 = np.sqrt(oe.contract('kc,kc->', l1_old - l1, l1_old - l1))
            rms_l2 = np.sqrt(oe.contract('klcd,klcd->', l2_old - l2, l2_old - l2))
            delta_E = E_pseudo_old - E_pseudo

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_pseudo.real, E_pseudo.imag, delta_E.real, delta_E.imag, rms_l1.real, rms_l1.imag, rms_l2.real, rms_l2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_l1 < self.parameters['d_convergence'] and rms_l2 < self.parameters['d_convergence']:
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_l2 > self.parameters['d_convergence']:
                    warnings.warn(
                        f"CCSD Lambda (SO) did not converge in {self.parameters['max_iterations']} iterations. "
                        f"|ΔE_pseudo| = {abs(delta_E):.2e}, RMS(L2) = {float(rms_l2.real):.2e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            iteration += 1

        return l1, l2



    def _lambda_residuals_SO(self, F, G, t1, t2, l1, l2, o_, v_):
        """
        Computes the CCSD Lambda singles and doubles residuals.

        Builds every intermediate and evaluates both residuals, mirroring
        ``_residuals_SO`` on the T side. The solver does not call this: with t
        fixed, all but the two G intermediates are invariants of the Lambda
        iteration, so ``_solve_lambda_SO`` builds them once and calls the two
        residual methods directly. This entry point exists for callers that want
        the residuals for one set of amplitudes without managing that hoisting.

        Returns
        -------
        r_L1 : ndarray
            Lambda singles residual, shape ``(no, nv)``.
        r_L2 : ndarray
            Lambda doubles residual, shape ``(no, no, nv, nv)``.
        """
        # F_ia is the T-side F_me read at (i,a); it is not a separate object.
        F_ia = self._F_me_SO(F, G, t1, o_, v_)

        F_ea = self._lambda_F_ea_SO(F, G, t1, t2, o_, v_)
        F_im = self._lambda_F_im_SO(F, G, t1, t2, o_, v_)
        W_eima = self._lambda_W_eima_SO(G, t1, t2, o_, v_)
        W_efma = self._lambda_W_efma_SO(F, G, t1, t2, o_, v_)
        W_eimn = self._lambda_W_eimn_SO(F, G, t1, t2, o_, v_)
        W_ijmn = self._lambda_W_ijmn_SO(G, t1, t2, o_, v_)
        W_efab = self._lambda_W_efab_SO(G, t1, t2, o_, v_)

        G_om = self._lambda_G_om_SO(t2, l2)
        G_ge = self._lambda_G_ge_SO(t2, l2)

        r_L1 = self._lambda_singles_residual_SO(G, t1, l1, l2, F_ia, F_ea, F_im,
                                                W_eima, W_efma, W_eimn,
                                                G_om, G_ge, o_, v_)
        r_L2 = self._lambda_doubles_residual_SO(G, t1, l1, l2, F_ia, F_ea, F_im,
                                                W_eima, W_ijmn, W_efab,
                                                G_om, G_ge, o_, v_)

        return r_L1, r_L2



    def _lambda_singles_residual_SO(self, G, t1, l1, l2, F_ia, F_ea, F_im,
                                    W_eima, W_efma, W_eimn, G_om, G_ge, o_, v_):
        """
        Computes the CCSD Lambda singles residual.

            0 = dL/dt_i^a = < Phi_0 | (1 + Lambda) [ H_bar tau_i^a ]_C | Phi_0 >

        Free indices are i (occupied) and a (virtual); dummy indices are m,n,o
        (occupied) and e,f,g (virtual).

            r_i^a = F_ia + F~_ea l_i^e - F~_im l_m^a + W~_eima l_m^e
                  + 1/2 W~_efma l_mi^ef - 1/2 W~_eimn l_mn^ea
                  + G_om (<oi||ma> + <oi||ga> t_m^g)
                  - G_ge (<ei||ga> - <oi||ga> t_o^e)

        Notes
        -----
        The driver is f_ia, carried in by F_ia, not f_ai: Lambda projects from the
        left, so its drivers are the bra-ket transposes of the T drivers.

        The t1*l1 piece of the two G groups is deliberately left inside the doubles
        residual rather than folded into G. Folding it in is legitimate, but the
        augmented G then leaves extra terms on this side that have to be carried
        explicitly; see the derivation notes.
        """
        r_L1  = F_ia.copy()

        # Terms carried by the one-particle intermediates.
        r_L1 += oe.contract('ea,ie->ia', F_ea, l1)
        r_L1 -= oe.contract('im,ma->ia', F_im, l1)

        # Terms carried by the two-particle intermediates.
        r_L1 += oe.contract('eima,me->ia', W_eima, l1)
        r_L1 += 0.5 * oe.contract('efma,mief->ia', W_efma, l2)
        r_L1 -= 0.5 * oe.contract('eimn,mnea->ia', W_eimn, l2)

        # The two Lambda-dependent G intermediates, each against a two-term bracket.
        r_L1 += oe.contract('om,oima->ia', G_om, G[o_,o_,o_,v_])
        r_L1 += oe.contract('om,oiga,mg->ia', G_om, G[o_,o_,v_,v_], t1)
        r_L1 -= oe.contract('ge,eiga->ia', G_ge, G[v_,o_,v_,v_])
        r_L1 += oe.contract('ge,oiga,oe->ia', G_ge, G[o_,o_,v_,v_], t1)

        return r_L1



    def _lambda_doubles_residual_SO(self, G, t1, l1, l2, F_ia, F_ea, F_im,
                                    W_eima, W_ijmn, W_efab, G_om, G_ge, o_, v_):
        """
        Computes the CCSD Lambda doubles residual.

            0 = dL/dt_ij^ab = < Phi_0 | (1 + Lambda) [ H_bar tau_ij^ab ]_C | Phi_0 >

        Free indices are i,j (occupied) and a,b (virtual); dummy indices are m,n
        (occupied) and e,f (virtual).

            r_ij^ab = <ij||ab>
                    + P(a,b) F~_eb l_ij^ae - P(i,j) F~_jm l_im^ab
                    + 1/2 W~_ijmn l_mn^ab + 1/2 W~_efab l_ij^ef
                    + P(i,j) P(a,b) W~_ejmb l_im^ae
                    + P(i,j) <ej||ab> l_i^e - P(a,b) <ij||mb> l_m^a
                    + P(i,j) P(a,b) F_jb l_i^a
                    - P(i,j) <mi||ab> (G_mj - t_m^e l_j^e)
                    + P(a,b) <ij||ae> (G_eb - t_m^e l_m^b)

        where P(i,j) X_ijab = X_ijab - X_jiab and P(a,b) X_ijab = X_ijab - X_ijba.

        Notes
        -----
        Six of the eight intermediates are the singles ones read at different
        indices: F~_eb and F~_jm are F~_ea and F~_im, W~_ejmb is W~_eima, F_jb is
        F_ia, and G_mj, G_eb are G_om, G_ge. Only W~_ijmn and W~_efab are new.

        The driver is <ij||ab>, not <ab||ij>; the two differ for complex orbitals.
        """
        # Terms already antisymmetric in both index pairs.
        r_L2  = G[o_,o_,v_,v_].copy()
        r_L2 += 0.5 * oe.contract('ijmn,mnab->ijab', W_ijmn, l2)
        r_L2 += 0.5 * oe.contract('efab,ijef->ijab', W_efab, l2)

        # The G groups keep their t1*l1 piece here rather than inside G, so that one
        # definition of G serves both residuals without extra correction terms.
        G_oo_dressed = G_om - oe.contract('me,je->mj', t1, l1)
        G_vv_dressed = G_ge - oe.contract('me,mb->eb', t1, l1)

        # Terms carrying P(a,b).
        r_P_ab  = oe.contract('eb,ijae->ijab', F_ea, l2)
        r_P_ab -= oe.contract('ijmb,ma->ijab', G[o_,o_,o_,v_], l1)
        r_P_ab += oe.contract('ijae,eb->ijab', G[o_,o_,v_,v_], G_vv_dressed)

        # Terms carrying P(i,j).
        r_P_ij  = -oe.contract('jm,imab->ijab', F_im, l2)
        r_P_ij += oe.contract('ejab,ie->ijab', G[v_,o_,v_,v_], l1)
        r_P_ij -= oe.contract('miab,mj->ijab', G[o_,o_,v_,v_], G_oo_dressed)

        # Terms carrying P(i,j) P(a,b).
        r_P_ij_ab  = oe.contract('ejmb,imae->ijab', W_eima, l2)
        r_P_ij_ab += oe.contract('jb,ia->ijab', F_ia, l1)

        # Apply each permutation operator once to the group that carries it.
        r_L2 += r_P_ab - r_P_ab.swapaxes(2,3)
        r_L2 += r_P_ij - r_P_ij.swapaxes(0,1)
        r_L2 += (r_P_ij_ab
                 - r_P_ij_ab.swapaxes(0,1)
                 - r_P_ij_ab.swapaxes(2,3)
                 + r_P_ij_ab.swapaxes(0,1).swapaxes(2,3))

        return r_L2



    def _lambda_F_ea_SO(self, F, G, t1, t2, o_, v_):
        """
        Computes the Lambda one-particle intermediate F~_ea.

            F~_ea = f_ea - f_na t_n^e + <ne||fa> t_n^f
                         - <no||fa> t_n^f t_o^e - 1/2 <no||fa> t_no^fe

        Depends on t alone, so it is an invariant of the Lambda iteration.
        """
        F_ea  = F[v_,v_].copy()
        F_ea -= oe.contract('na,ne->ea', F[o_,v_], t1)
        F_ea += oe.contract('nefa,nf->ea', G[o_,v_,v_,v_], t1)
        F_ea -= oe.contract('nofa,nf,oe->ea', G[o_,o_,v_,v_], t1, t1)
        F_ea -= 0.5 * oe.contract('nofa,nofe->ea', G[o_,o_,v_,v_], t2)

        return F_ea



    def _lambda_F_im_SO(self, F, G, t1, t2, o_, v_):
        """
        Computes the Lambda one-particle intermediate F~_im.

            F~_im = f_im + f_if t_m^f - <ni||mf> t_n^f
                         + <ni||fg> t_n^f t_m^g + 1/2 <ni||fg> t_nm^fg

        Depends on t alone, so it is an invariant of the Lambda iteration.
        """
        F_im  = F[o_,o_].copy()
        F_im += oe.contract('if,mf->im', F[o_,v_], t1)
        F_im -= oe.contract('nimf,nf->im', G[o_,o_,o_,v_], t1)
        F_im += oe.contract('nifg,nf,mg->im', G[o_,o_,v_,v_], t1, t1)
        F_im += 0.5 * oe.contract('nifg,nmfg->im', G[o_,o_,v_,v_], t2)

        return F_im



    def _lambda_W_eima_SO(self, G, t1, t2, o_, v_):
        """
        Computes the Lambda two-particle intermediate W~_eima.

            W~_eima = <ei||ma> - <ni||ma> t_n^e + <ei||fa> t_m^f
                              + <ni||fa> t_nm^fe - <ni||fa> t_n^e t_m^f

        Both residuals use this one: the doubles reads it at (e,j,m,b).
        """
        W_eima  = G[v_,o_,o_,v_].copy()
        W_eima -= oe.contract('nima,ne->eima', G[o_,o_,o_,v_], t1)
        W_eima += oe.contract('eifa,mf->eima', G[v_,o_,v_,v_], t1)
        W_eima += oe.contract('nifa,nmfe->eima', G[o_,o_,v_,v_], t2)
        W_eima -= oe.contract('nifa,ne,mf->eima', G[o_,o_,v_,v_], t1, t1)

        return W_eima



    def _lambda_W_efma_SO(self, F, G, t1, t2, o_, v_):
        """
        Computes the Lambda two-particle intermediate W~_efma.

            W~_efma = <ef||ma> + f_na t_nm^ef + <ef||ga> t_m^g
                    + 2 <ne||ma> t_n^f + <no||ma> t_n^e t_o^f
                    - 2 <ne||ga> t_nm^gf + 2 <ne||ga> t_n^f t_m^g
                    + 1/2 <no||ma> t_no^ef - <no||ga> t_nm^ef t_o^g
                    + <no||ga> t_n^e t_o^f t_m^g - 2 <no||ga> t_nm^ge t_o^f
                    + 1/2 <no||ga> t_no^ef t_m^g

        Used only by the singles residual. It meets l_mi^ef under a 1/2, so only
        its (e,f)-antisymmetric part contributes; the factor-2 terms above are the
        collected form and are not themselves antisymmetric in e and f.
        """
        W_efma  = G[v_,v_,o_,v_].copy()
        W_efma += oe.contract('na,nmef->efma', F[o_,v_], t2)
        W_efma += oe.contract('efga,mg->efma', G[v_,v_,v_,v_], t1)
        W_efma += 2 * oe.contract('nema,nf->efma', G[o_,v_,o_,v_], t1)
        W_efma += oe.contract('noma,ne,of->efma', G[o_,o_,o_,v_], t1, t1)
        W_efma -= 2 * oe.contract('nega,nmgf->efma', G[o_,v_,v_,v_], t2)
        W_efma += 2 * oe.contract('nega,nf,mg->efma', G[o_,v_,v_,v_], t1, t1)
        W_efma += 0.5 * oe.contract('noma,noef->efma', G[o_,o_,o_,v_], t2)
        W_efma -= oe.contract('noga,nmef,og->efma', G[o_,o_,v_,v_], t2, t1)
        W_efma += oe.contract('noga,ne,of,mg->efma', G[o_,o_,v_,v_], t1, t1, t1)
        W_efma -= 2 * oe.contract('noga,nmge,of->efma', G[o_,o_,v_,v_], t2, t1)
        W_efma += 0.5 * oe.contract('noga,noef,mg->efma', G[o_,o_,v_,v_], t2, t1)

        return W_efma



    def _lambda_W_eimn_SO(self, F, G, t1, t2, o_, v_):
        """
        Computes the Lambda two-particle intermediate W~_eimn.

            W~_eimn = <ei||mn> - f_if t_mn^fe - <oi||mn> t_o^e
                    + 2 <ei||mf> t_n^f + <ei||fg> t_m^f t_n^g
                    + 2 <oi||mf> t_on^fe - 2 <oi||mf> t_o^e t_n^f
                    + 1/2 <ei||fg> t_mn^fg - <oi||fg> t_o^f t_mn^ge
                    - <oi||fg> t_o^e t_m^f t_n^g + 2 <oi||fg> t_om^fe t_n^g
                    - 1/2 <oi||fg> t_o^e t_mn^fg

        Used only by the singles residual. It meets l_mn^ea under a 1/2, so only
        its (m,n)-antisymmetric part contributes; the factor-2 terms above are the
        collected form and are not themselves antisymmetric in m and n.
        """
        W_eimn  = G[v_,o_,o_,o_].copy()
        W_eimn -= oe.contract('if,mnfe->eimn', F[o_,v_], t2)
        W_eimn -= oe.contract('oimn,oe->eimn', G[o_,o_,o_,o_], t1)
        W_eimn += 2 * oe.contract('eimf,nf->eimn', G[v_,o_,o_,v_], t1)
        W_eimn += oe.contract('eifg,mf,ng->eimn', G[v_,o_,v_,v_], t1, t1)
        W_eimn += 2 * oe.contract('oimf,onfe->eimn', G[o_,o_,o_,v_], t2)
        W_eimn -= 2 * oe.contract('oimf,oe,nf->eimn', G[o_,o_,o_,v_], t1, t1)
        W_eimn += 0.5 * oe.contract('eifg,mnfg->eimn', G[v_,o_,v_,v_], t2)
        W_eimn -= oe.contract('oifg,of,mnge->eimn', G[o_,o_,v_,v_], t1, t2)
        W_eimn -= oe.contract('oifg,oe,mf,ng->eimn', G[o_,o_,v_,v_], t1, t1, t1)
        W_eimn += 2 * oe.contract('oifg,omfe,ng->eimn', G[o_,o_,v_,v_], t2, t1)
        W_eimn -= 0.5 * oe.contract('oifg,oe,mnfg->eimn', G[o_,o_,v_,v_], t1, t2)

        return W_eimn



    def _lambda_W_ijmn_SO(self, G, t1, t2, o_, v_):
        """
        Computes the Lambda two-particle intermediate W~_ijmn.

            W~_ijmn = <ij||mn> + P(m,n) <ij||me> t_n^e
                    + <ij||ef> t_m^e t_n^f + 1/2 <ij||ef> t_mn^ef

        where P(m,n) X_ijmn = X_ijmn - X_ijnm.

        Notes
        -----
        The singles term carries P(m,n), not a factor of two. Both contract to the
        same number against the antisymmetric l_mn^ab, but only the permuted form
        has the antisymmetry of the block it represents.
        """
        W_ijmn  = G[o_,o_,o_,o_].copy()
        P_mn = oe.contract('ijme,ne->ijmn', G[o_,o_,o_,v_], t1)
        W_ijmn += P_mn - P_mn.swapaxes(2,3)
        W_ijmn += oe.contract('ijef,me,nf->ijmn', G[o_,o_,v_,v_], t1, t1)
        W_ijmn += 0.5 * oe.contract('ijef,mnef->ijmn', G[o_,o_,v_,v_], t2)

        return W_ijmn



    def _lambda_W_efab_SO(self, G, t1, t2, o_, v_):
        """
        Computes the Lambda two-particle intermediate W~_efab.

            W~_efab = <ef||ab> + P(e,f) <me||ab> t_m^f
                    + <mn||ab> t_m^e t_n^f + 1/2 <mn||ab> t_mn^ef

        where P(e,f) X_efab = X_efab - X_feab.

        Notes
        -----
        As for W~_ijmn, the singles term carries P(e,f) rather than a factor of two
        so that the stored tensor is antisymmetric in e and f.
        """
        W_efab  = G[v_,v_,v_,v_].copy()
        P_ef = oe.contract('meab,mf->efab', G[o_,v_,v_,v_], t1)
        W_efab += P_ef - P_ef.swapaxes(0,1)
        W_efab += oe.contract('mnab,me,nf->efab', G[o_,o_,v_,v_], t1, t1)
        W_efab += 0.5 * oe.contract('mnab,mnef->efab', G[o_,o_,v_,v_], t2)

        return W_efab



    def _lambda_G_om_SO(self, t2, l2):
        """
        Computes the Lambda-dependent intermediate G_om.

            G_om = 1/2 t_on^fg l_nm^fg

        This one depends on Lambda, so it is rebuilt every iteration. The doubles
        residual reads it at (m,j).
        """
        return 0.5 * oe.contract('onfg,nmfg->om', t2, l2)



    def _lambda_G_ge_SO(self, t2, l2):
        """
        Computes the Lambda-dependent intermediate G_ge.

            G_ge = 1/2 t_no^gf l_no^fe

        This one depends on Lambda, so it is rebuilt every iteration. The doubles
        residual reads it at (e,b).
        """
        return 0.5 * oe.contract('nogf,nofe->ge', t2, l2)
