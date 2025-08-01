"""Contains the configuration interaction doubles (CID) wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
import opt_einsum as oe
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import solve_DIIS
from apyib.mp2_wfn import mp2_wfn
from apyib.utils import get_slices
from apyib.utils import compute_F_MO
from apyib.utils import compute_ERI_MO
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO
from apyib.utils import solve_general_DIIS

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)
class ci_wfn(object):
    """
    Wavefunction object.
    """
    # Define the specific properties of the CI wavefunction.
    def __init__(self, parameters, wfn):

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



    def solve_CID(self, print_level=0):
        """
        Solves the CID amplitudes and energies.
        """
        # Setting up slice options for the MO integrals.
        o_ = self.I_list[1]
        v_ = self.I_list[2]

        # For readability.
        F_MO = self.F_MO.copy()
        ERI_MO = self.ERI_MO.copy()

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)       # (pr|qs) -> <pq|rs>

        # Initial T2 guess amplitude.
        t2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] / self.D_ijab

        # Solve for initial CID energy.
        E_CID =  oe.contract('ijab,ijab->', 2 * ERI_MO[o_,o_,v_,v_] - ERI_MO.swapaxes(2,3)[o_,o_,v_,v_], t2)
        t2 = t2.copy()

        if print_level > 0:
            print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_T2(real)      RMS_T2(imaginary)")

        # Start iterative procedure.
        iteration = 1 
        while iteration <= self.parameters['max_iterations']:
            E_CID_old = E_CID
            t2_old = t2.copy()

            # Solving for the residual.
            r_T2 = 0.5 * ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
            r_T2 += oe.contract('ijae,be->ijab', t2, F_MO[v_,v_])
            r_T2 -= oe.contract('imab,mj->ijab', t2, F_MO[o_,o_])
            r_T2 += 0.5 * oe.contract('mnab,mnij->ijab', t2, ERI_MO[o_,o_,o_,o_])
            r_T2 += 0.5 * oe.contract('ijef,abef->ijab', t2, ERI_MO[v_,v_,v_,v_])
            r_T2 += oe.contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), ERI_MO[o_,v_,v_,o_])
            r_T2 += oe.contract('imae,mbej->ijab', t2, (ERI_MO[o_,v_,v_,o_] - ERI_MO.swapaxes(2,3)[o_,v_,v_,o_]))
            r_T2 -= oe.contract('mjae,mbie->ijab', t2, ERI_MO[o_,v_,o_,v_])

            r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
            r_T2 -= E_CID * t2
            t2 += r_T2 / self.D_ijab

            # Perform DIIS extrapolation.
            if self.parameters['DIIS']:
                occ = len(t2)
                vir = len(t2[0][0])
                t2_flat = len(np.reshape(t2, (-1)))
                res_vec = np.reshape(r_T2, (-1))
                t_vec = np.reshape(t2, (-1))
                if iteration == 1:
                    t_iter = np.atleast_2d(t_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                t2 = np.reshape(t_vec, (occ, occ, vir, vir))

            # Compute new CID energy.
            E_CID = oe.contract('ijab,ijab->', 2 * ERI_MO[o_,o_,v_,v_] - ERI_MO.swapaxes(2,3)[o_,o_,v_,v_], t2)

            # Compute total energy.
            E_tot = self.wfn.E_SCF + E_CID + self.H.E_nuc

            # Compute convergence data.
            rms_t2 = oe.contract('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CID_old - E_CID

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CID.real, E_CID.imag, E_tot, delta_E.real, delta_E.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        return E_CID, t2



    def solve_CID_SO(self, print_level=0):
        """ 
        Solves the CID amplitudes and energies.
        """
        # Build energy denominators in the spin orbital basis.
        eps_o = np.repeat(self.eps_o, 2)
        eps_v = np.repeat(self.eps_v, 2)
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

        # Initial T2 guess amplitude.
        t2 = (ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] - ERI_SO.copy().swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]) / D_ijab

        # Solve for initial CID energy.
        E_CID =  0.25 * oe.contract('ijab,ijab->', t2, ERI_SO[o_,o_,v_,v_] - ERI_SO.swapaxes(2,3)[o_,o_,v_,v_])
        t2 = t2.copy()

        if print_level > 0:
            print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_T2(real)      RMS_T2(imaginary)")

        # Start iterative procedure.
        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_CID_old = E_CID
            t2_old = t2.copy()

            # Solving for the residual. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            r_T2 = ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] - ERI_SO.copy().swapaxes(0,2).swapaxes(1,3).swapaxes(2,3)[o_,o_,v_,v_]
            r_T2 += oe.contract('ijae,be->ijab', t2, F_SO[v_,v_]) + oe.contract('ijeb,ae->ijab', t2, F_SO[v_,v_])
            r_T2 -= oe.contract('imab,mj->ijab', t2, F_SO[o_,o_]) + oe.contract('mjab,mi->ijab', t2, F_SO[o_,o_])
            r_T2 += 0.5 * oe.contract('mnab,mnij->ijab', t2, ERI_SO[o_,o_,o_,o_] - ERI_SO.swapaxes(2,3)[o_,o_,o_,o_])
            r_T2 += 0.5 * oe.contract('ijef,abef->ijab', t2, ERI_SO[v_,v_,v_,v_] - ERI_SO.swapaxes(2,3)[v_,v_,v_,v_])
            r_T2 += oe.contract('imae,mbej->ijab', t2, ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_])
            r_T2 += oe.contract('mjae,mbei->ijab', t2, ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_])
            r_T2 += oe.contract('imeb,maej->ijab', t2, ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_])
            r_T2 += oe.contract('mjeb,maei->ijab', t2, ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_])                     

            r_T2 -= E_CID * t2
            t2 += r_T2 / D_ijab

            # Perform DIIS extrapolation.
            if self.parameters['DIIS']:
                occ = len(t2)
                vir = len(t2[0][0])
                t2_flat = len(np.reshape(t2, (-1)))
                res_vec = np.reshape(r_T2, (-1))
                t_vec = np.reshape(t2, (-1))
                if iteration == 1:
                    t_iter = np.atleast_2d(t_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                t2 = np.reshape(t_vec, (occ, occ, vir, vir))

            # Compute new CID energy.
            E_CID = 0.25 * oe.contract('ijab,ijab->', ERI_SO[o_,o_,v_,v_] - ERI_SO.swapaxes(2,3)[o_,o_,v_,v_], t2)

            # Compute total energy.
            E_tot = self.wfn.E_SCF + E_CID + self.H.E_nuc

            # Compute convergence data.
            rms_t2 = oe.contract('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CID_old - E_CID

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CID.real, E_CID.imag, E_tot, delta_E.real, delta_E.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        return E_CID, t2



    def solve_CISD_SO(self, print_level=0):
        """ 
        Solves the CISD amplitudes and energies.
        """
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

        # Initial T1 guess amplitude.
        t1 = F_SO.copy().swapaxes(0,1)[o_,v_] / D_ia

        # Initial T2 guess amplitude.
        t2 = (ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] - ERI_SO.copy().swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]) / D_ijab

        # Solve for initial CID energy.
        E_CISD =  oe.contract('ia,ia->', t1, F_SO[o_,v_] ) + 0.25 * oe.contract('ijab,ijab->', t2, ERI_SO[o_,o_,v_,v_] - ERI_SO.swapaxes(2,3)[o_,o_,v_,v_])
        t1 = t1.copy()
        t2 = t2.copy()

        if print_level > 0:
            print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_T1(real)      RMS_T1(imaginary)      RMS_T2(real)      RMS_T2(imaginary)")

        # Start iterative procedure.
        iteration = 1 
        while iteration <= self.parameters['max_iterations']:
            E_CISD_old = E_CISD
            t1_old = t1.copy()
            t2_old = t2.copy()

            # Solving for the residuals. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            #r_T1 = F_SO[0:o,o:nbf].copy()
            r_T1 = F_SO.copy().swapaxes(0,1)[o_,v_]
            r_T1 -= oe.contract('ji,ja->ia', F_SO[o_,o_], t1)
            r_T1 += oe.contract('ab,ib->ia', F_SO[v_,v_], t1)
            r_T1 += oe.contract('jabi,jb->ia', ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_], t1)
            r_T1 += oe.contract('jb,ijab->ia', F_SO[o_,v_], t2)
            r_T1 += 0.5 * oe.contract('ajcb,ijcb->ia', ERI_SO[v_,o_,v_,v_] - ERI_SO.swapaxes(2,3)[v_,o_,v_,v_], t2)
            r_T1 -= 0.5 * oe.contract('kjib,kjab->ia', ERI_SO[o_,o_,o_,v_] - ERI_SO.swapaxes(2,3)[o_,o_,o_,v_], t2)
            r_T1 -= E_CISD * t1

            #r_T2 = ERI_SO[0:o,0:o,o:nbf,o:nbf].copy() - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf].copy()
            r_T2 = ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] - ERI_SO.copy().swapaxes(0,2).swapaxes(1,3).swapaxes(2,3)[o_,o_,v_,v_]
            r_T2 -= oe.contract('kbij,ka->ijab', ERI_SO[o_,v_,o_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,o_,o_], t1)
            r_T2 -= oe.contract('akij,kb->ijab', ERI_SO[v_,o_,o_,o_] - ERI_SO.swapaxes(2,3)[v_,o_,o_,o_], t1)
            r_T2 += oe.contract('abcj,ic->ijab', ERI_SO[v_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[v_,v_,v_,o_], t1)
            r_T2 += oe.contract('abic,jc->ijab', ERI_SO[v_,v_,o_,v_] - ERI_SO.swapaxes(2,3)[v_,v_,o_,v_], t1)
            r_T2 += oe.contract('bc,ijac->ijab', F_SO[v_,v_], t2)
            r_T2 += oe.contract('ac,ijcb->ijab', F_SO[v_,v_], t2)
            r_T2 -= oe.contract('kj,ikab->ijab', F_SO[o_,o_], t2)
            r_T2 -= oe.contract('ki,kjab->ijab', F_SO[o_,o_], t2)
            r_T2 += 0.5 * oe.contract('klij,klab->ijab', ERI_SO[o_,o_,o_,o_] - ERI_SO.swapaxes(2,3)[o_,o_,o_,o_], t2)
            r_T2 += 0.5 * oe.contract('abcd,ijcd->ijab', ERI_SO[v_,v_,v_,v_] - ERI_SO.swapaxes(2,3)[v_,v_,v_,v_], t2)
            r_T2 += oe.contract('kbcj,ikac->ijab', ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_], t2)
            r_T2 += oe.contract('kbci,kjac->ijab', ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_], t2)
            r_T2 += oe.contract('kacj,ikcb->ijab', ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_], t2)
            r_T2 += oe.contract('kaci,kjcb->ijab', ERI_SO[o_,v_,v_,o_] - ERI_SO.swapaxes(2,3)[o_,v_,v_,o_], t2)
            r_T2 -= E_CISD * t2

            t1 += r_T1 /D_ia
            t2 += r_T2 / D_ijab

            # Perform DIIS extrapolation.
            if self.parameters['DIIS']:
                occ = len(t1)
                vir = len(t1[0])
                t1_flat = len(np.reshape(t1, (-1)))
                t2_flat = len(np.reshape(t2, (-1)))
                res_vec = np.concatenate((np.reshape(r_T1, (-1)), np.reshape(r_T2, (-1))))
                t_vec = np.concatenate((np.reshape(t1, (-1)), np.reshape(t2, (-1))))
                if iteration == 1:
                    t_iter = np.atleast_2d(t_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                t1 = np.reshape(t_vec[0:t1_flat], (occ, vir))
                t2 = np.reshape(t_vec[t1_flat:], (occ, occ, vir, vir))

            # Compute new CISD energy.
            E_CISD = oe.contract('ia,ia->', t1, F_SO[o_,v_]) + 0.25 * oe.contract('ijab,ijab->', t2, ERI_SO[o_,o_,v_,v_] - ERI_SO.swapaxes(2,3)[o_,o_,v_,v_])

            # Compute total energy.
            E_tot = self.wfn.E_SCF + E_CISD + self.H.E_nuc

            # Compute convergence data.
            rms_t1 = oe.contract('ia,ia->', t1_old - t1, t1_old - t1)
            rms_t1 = np.sqrt(rms_t1)

            rms_t2 = oe.contract('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CISD_old - E_CISD

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CISD.real, E_CISD.imag, E_tot, delta_E.real, delta_E.imag, rms_t1.real, rms_t1.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t1 < self.parameters['d_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        #### Testing adjoint formulation of CISD equations. ####

        ## Compute and normalize amplitudes.
        #N = 1 / np.sqrt(1**2 + oe.contract('ia,ia->', np.conjugate(t1), t1) + 0.25 * oe.contract('ijab,ijab->', np.conjugate(t2), t2))
        #t0 = N
        #t1 *= N
        #t2 *= N

        ## Build OPD.
        #D_pq = np.zeros_like(F_SO)
        #D_pq[o_,o_] -= oe.contract('ja,ia->ij', np.conjugate(t1), t1) + 0.5 * oe.contract('jkab,ikab->ij', np.conjugate(t2), t2)
        #D_pq[v_,v_] += oe.contract('ia,ib->ab', np.conjugate(t1), t1) + 0.5 * oe.contract('ijac,ijbc->ab', np.conjugate(t2), t2)
        #D_pq[o_,v_] += np.conjugate(t0) * t1 + oe.contract('jb,ijab->ia', np.conjugate(t1), t2)
        #D_pq[v_,o_] += np.conjugate(t1.T) * t0 + oe.contract('ijab,jb->ai', np.conjugate(t2), t1)

        ## Build TPD.
        #D_pqrs = np.zeros_like(ERI_SO)
        #D_pqrs[o_,o_,o_,o_] += 0.25 * oe.contract('klab,ijab->ijkl', np.conjugate(t2), t2)
        #D_pqrs[v_,v_,v_,v_] += 0.25 * oe.contract('ijab,ijcd->abcd', np.conjugate(t2), t2)
        #D_pqrs[o_,v_,v_,o_] += oe.contract('ja,ib->iabj', np.conjugate(t1), t1)
        #D_pqrs[o_,v_,o_,v_] -= oe.contract('ja,ib->iajb', np.conjugate(t1), t1)
        #D_pqrs[v_,o_,o_,v_] += oe.contract('jkac,ikbc->aijb', np.conjugate(t2), t2)
        #D_pqrs[v_,o_,v_,o_] -= oe.contract('jkac,ikbc->aibj', np.conjugate(t2), t2)
        #D_pqrs[o_,o_,v_,v_] += 0.5 * np.conjugate(t0) * t2
        #D_pqrs[v_,v_,o_,o_] += 0.5 * np.conjugate(t2.swapaxes(0,2).swapaxes(1,3)) * t0
        #D_pqrs[v_,o_,v_,v_] += oe.contract('ja,ijcb->aibc', np.conjugate(t1), t2)
        #D_pqrs[o_,v_,o_,o_] -= oe.contract('kjab,ib->iajk', np.conjugate(t2), t1)
        #D_pqrs[v_,v_,v_,o_] += oe.contract('jiab,jc->abci', np.conjugate(t2), t1)
        #D_pqrs[o_,o_,o_,v_] -= oe.contract('kb,ijba->ijka', np.conjugate(t1), t2)

        ## Compute energy.
        #E = oe.contract('pq,pq->', F_SO, D_pq) + oe.contract('pqrs,pqrs->', ERI_SO, D_pqrs)
        #print("Adjoint Correlation Energy: ", E)

        ## Equations for adjoint formulation are correct.

        return E_CISD, t1, t2



    def solve_CISD(self, print_level=0):
        """ 
        Solves the CISD amplitudes and energies.
        """
        # Setting up slice options for the MO integrals.
        o_ = self.I_list[1]
        v_ = self.I_list[2]

        # For readability.
        F_MO = self.F_MO.copy()
        ERI_MO = self.ERI_MO.copy()

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)       # (pr|qs) -> <pq|rs>

        # Initial T1 guess amplitude.
        t1 = F_MO.copy().swapaxes(0,1)[o_,v_] / self.D_ia

        # Initial T2 guess amplitude.
        t2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] / self.D_ijab

        # Solve for initial CISD energy.
        E_CISD =  2.0 * oe.contract('ia,ia->', t1, F_MO[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * ERI_MO[o_,o_,v_,v_] - ERI_MO.swapaxes(2,3)[o_,o_,v_,v_])
        t1 = t1.copy()
        t2 = t2.copy()

        if print_level > 0:
            print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_T1(real)      RMS_T1(imaginary)      RMS_T2(real)      RMS_T2(imaginary)")

        # Start iterative procedure.
        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_CISD_old = E_CISD
            t1_old = t1.copy()
            t2_old = t2.copy()

            # Solving for the residuals. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            r_T1 = F_MO.copy().swapaxes(0,1)[o_,v_]
            r_T1 -= oe.contract('ji,ja->ia', F_MO[o_,o_], t1)
            r_T1 += oe.contract('ab,ib->ia', F_MO[v_,v_], t1)
            r_T1 += oe.contract('jabi,jb->ia', 2.0 * ERI_MO[o_,v_,v_,o_] - ERI_MO.swapaxes(2,3)[o_,v_,v_,o_], t1)
            r_T1 += oe.contract('jb,ijab->ia', F_MO[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
            r_T1 += oe.contract('ajbc,ijbc->ia', 2.0 * ERI_MO[v_,o_,v_,v_] - ERI_MO.swapaxes(2,3)[v_,o_,v_,v_], t2)
            r_T1 -= oe.contract('kjib,kjab->ia', 2.0 * ERI_MO[o_,o_,o_,v_] - ERI_MO.swapaxes(2,3)[o_,o_,o_,v_], t2)
            r_T1 -= E_CISD * t1

            r_T2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
            r_T2 += oe.contract('abcj,ic->ijab', ERI_MO[v_,v_,v_,o_], t1)
            r_T2 += oe.contract('abic,jc->ijab', ERI_MO[v_,v_,o_,v_], t1)
            r_T2 -= oe.contract('kbij,ka->ijab', ERI_MO[o_,v_,o_,o_], t1)
            r_T2 -= oe.contract('akij,kb->ijab', ERI_MO[v_,o_,o_,o_], t1)
            r_T2 += oe.contract('ac,ijcb->ijab', F_MO[v_,v_], t2)
            r_T2 += oe.contract('bc,ijac->ijab', F_MO[v_,v_], t2)
            r_T2 -= oe.contract('ki,kjab->ijab', F_MO[o_,o_], t2)
            r_T2 -= oe.contract('kj,ikab->ijab', F_MO[o_,o_], t2)
            r_T2 += oe.contract('klij,klab->ijab', ERI_MO[o_,o_,o_,o_], t2)
            r_T2 += oe.contract('abcd,ijcd->ijab', ERI_MO[v_,v_,v_,v_], t2)            
            r_T2 -= oe.contract('kbcj,ikca->ijab', ERI_MO[o_,v_,v_,o_], t2)
            r_T2 += oe.contract('kaci,kjcb->ijab', 2.0 * ERI_MO[o_,v_,v_,o_] - ERI_MO.swapaxes(2,3)[o_,v_,v_,o_], t2)
            r_T2 -= oe.contract('kbic,kjac->ijab', ERI_MO[o_,v_,o_,v_], t2)
            r_T2 -= oe.contract('kaci,kjbc->ijab', ERI_MO[o_,v_,v_,o_], t2)
            r_T2 += oe.contract('kbcj,ikac->ijab', 2.0 * ERI_MO[o_,v_,v_,o_] - ERI_MO.swapaxes(2,3)[o_,v_,v_,o_], t2)
            r_T2 -= oe.contract('kajc,ikcb->ijab', ERI_MO[o_,v_,o_,v_], t2)
            r_T2 -= E_CISD * t2

            t1 += r_T1 / self.D_ia
            t2 += r_T2 / self.D_ijab

            # Perform DIIS extrapolation.
            if self.parameters['DIIS']:
                occ = len(t1)
                vir = len(t1[0])
                t1_flat = len(np.reshape(t1, (-1)))
                t2_flat = len(np.reshape(t2, (-1)))
                res_vec = np.concatenate((np.reshape(r_T1, (-1)), np.reshape(r_T2, (-1))))
                t_vec = np.concatenate((np.reshape(t1, (-1)), np.reshape(t2, (-1))))
                if iteration == 1:
                    t_iter = np.atleast_2d(t_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                t_vec, e_iter, t_iter = solve_general_DIIS(self.parameters, res_vec, t_vec, e_iter, t_iter, iteration)
                t1 = np.reshape(t_vec[0:t1_flat], (occ, vir))
                t2 = np.reshape(t_vec[t1_flat:], (occ, occ, vir, vir))

            # Compute new CISD energy.
            E_CISD =  2.0 * oe.contract('ia,ia->', t1, F_MO[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * ERI_MO[o_,o_,v_,v_] - ERI_MO.swapaxes(2,3)[o_,o_,v_,v_])

            # Compute total energy.
            E_tot = self.wfn.E_SCF + self.H.E_nuc + E_CISD

            # Compute convergence data.
            rms_t1 = oe.contract('ia,ia->', t1_old - t1, t1_old - t1)
            rms_t1 = np.sqrt(rms_t1)

            rms_t2 = oe.contract('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CISD_old - E_CISD

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CISD.real, E_CISD.imag, E_tot, delta_E.real, delta_E.imag, rms_t1.real, rms_t1.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t1 < self.parameters['d_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        #### Testing adjoint formulation of CISD equations. ####

        ## Compute and normalize amplitudes.
        #N = 1 / np.sqrt(1**2 + 2*oe.contract('ia,ia->', np.conjugate(t1), t1) + oe.contract('ijab,ijab->', np.conjugate(t2), 2*t2-t2.swapaxes(2,3)))
        #t0 = N
        #t1 *= N
        #t2 *= N

        ## Build OPD.
        #D_pq = np.zeros_like(F_MO)
        #D_pq[o_,o_] -= 2 * oe.contract('ja,ia->ij', np.conjugate(t1), t1) + 2 * oe.contract('jkab,ikab->ij', np.conjugate(2*t2 - t2.swapaxes(2,3)), t2)
        #D_pq[v_,v_] += 2 * oe.contract('ia,ib->ab', np.conjugate(t1), t1) + 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2*t2 - t2.swapaxes(2,3)), t2)
        #D_pq[o_,v_] += 2 * np.conjugate(t0) * t1 + 2 * oe.contract('jb,ijab->ia', np.conjugate(t1), t2 - t2.swapaxes(2,3))
        #D_pq[v_,o_] += 2 * np.conjugate(t1.T) * t0 + 2 * oe.contract('ijab,jb->ai', np.conjugate(t2 - t2.swapaxes(2,3)), t1)

        ## Build TPD.
        #D_pqrs = np.zeros_like(ERI_MO)
        #D_pqrs[o_,o_,o_,o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2), (2*t2 - t2.swapaxes(2,3))) 
        #D_pqrs[v_,v_,v_,v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2), (2*t2 - t2.swapaxes(2,3))) 
        #D_pqrs[o_,v_,v_,o_] += 4 * oe.contract('ja,ib->iabj', np.conjugate(t1), t1) 
        #D_pqrs[o_,v_,o_,v_] -= 2 * oe.contract('ja,ib->iajb', np.conjugate(t1), t1) 
        #D_pqrs[v_,o_,o_,v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2*t2 - t2.swapaxes(2,3)), 2*t2 - t2.swapaxes(2,3))
 
        #D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2), t2)
        #D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2), t2)
        #D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2), t2)
        #D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2), t2)

        #D_pqrs[o_,o_,v_,v_] += np.conjugate(t0) * (2*t2 -t2.swapaxes(2,3))
        #D_pqrs[v_,v_,o_,o_] += np.conjugate(2*t2.swapaxes(0,2).swapaxes(1,3) - t2.swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)) * t0
        #D_pqrs[v_,o_,v_,v_] += 2 * oe.contract('ja,ijcb->aibc', np.conjugate(t1), 2*t2 - t2.swapaxes(2,3))
        #D_pqrs[o_,v_,o_,o_] -= 2 * oe.contract('kjab,ib->iajk', np.conjugate(2*t2 - t2.swapaxes(2,3)), t1) 
        #D_pqrs[v_,v_,v_,o_] += 2 * oe.contract('jiab,jc->abci', np.conjugate(2*t2 - t2.swapaxes(2,3)), t1)
        #D_pqrs[o_,o_,o_,v_] -= 2 * oe.contract('kb,ijba->ijka', np.conjugate(t1), 2*t2 - t2.swapaxes(2,3))

        ## Compute energy.
        #E = oe.contract('pq,pq->', F_MO, D_pq) + oe.contract('pqrs,pqrs->', ERI_MO, D_pqrs)
        #print("Adjoint Correlation Energy: ", E)

        ## Equations for adjoint formulation are correct.

        return E_CISD, t1, t2

