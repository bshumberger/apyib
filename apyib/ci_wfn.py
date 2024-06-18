"""Contains the configuration interaction doubles (CID) wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import solve_DIIS
from apyib.mp2_wfn import mp2_wfn
from apyib.utils import compute_F_MO
from apyib.utils import compute_ERI_MO
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)
class ci_wfn(object):
    """
    Wavefunction object.
    """
    # Define the specific properties of the CI wavefunction.
    def __init__(self, parameters, E_SCF, E_tot, C):

        # Define the Hamiltonian and the Hartree-Fock reference energy and wavefunction.
        self.parameters = parameters
        self.H = Hamiltonian(parameters)
        self.wfn = hf_wfn(self.H)
        #self.e, self.E_SCF, self.E_tot, self.C = self.wfn.solve_SCF(parameters)
        self.E_SCF = E_SCF
        self.E_tot = E_tot
        self.C = C

        # Define the number of occupied and virtual orbitals.
        self.nbf = self.wfn.nbf
        self.no = self.wfn.ndocc
        self.nv = self.wfn.nbf - self.wfn.ndocc

        # For readability.
        nbf = self.nbf
        o = self.no
        v = self.nv

        # Set up MO one- and two-electron integrals.
        self.F_MO = compute_F_MO(self.parameters, self.H, self.wfn, self.C)
        self.ERI_MO = compute_ERI_MO(self.parameters, self.H, self.wfn, self.C)

        # Compute the SCF energy.
        H_core_MO = np.einsum('ip,ij,jq->pq', np.conjugate(self.C), self.H.T + self.H.V, self.C)
    
        E = 0.0 
        for i in range(0,self.no):
            E += H_core_MO[i][i] + self.F_MO[i][i]
        #print('Total Energy from SCF in CID Code:', E + self.H.E_nuc)


    def solve_CID(self):
        """
        Solves the CID amplitudes and energies.
        """

        # For readability.
        nbf = self.nbf
        o = self.no
        v = self.nv
        F_MO = self.F_MO.copy()
        ERI_MO = self.ERI_MO.copy()

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the numerators for the T2 guess amplitudes.
        t2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[0:o,0:o,o:nbf,o:nbf]

        # Set up the denominators for the T2 guess amplitudes.
        Dijab = np.ones_like(ERI_MO)
        for i in range(0,o):
            for j in range(0,o):
                for a in range(o,nbf):
                    for b in range(o,nbf):
                        Dijab[i][j][a][b] *= F_MO[i][i] + F_MO[j][j] - F_MO[a][a] - F_MO[b][b]
        Dijab = Dijab[0:o,0:o,o:nbf,o:nbf]

        # Initial T2 guess amplitude.
        t2 = t2 / Dijab

        # Solve for initial CID energy.
        E_CID =  np.einsum('ijab,ijab->', t2, 2 * ERI_MO[0:o,0:o,o:nbf,o:nbf] - ERI_MO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])
        t2 = t2.copy()
        #print(E_CID)

        # Start iterative procedure.
        iteration = 1 
        while iteration <= self.parameters['max_iterations']:
            E_CID_old = E_CID
            t2_old = t2.copy()

            # Solving for the residual.
            r_T2 = 0.5 * ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[0:o,0:o,o:nbf,o:nbf]                                                     # First term of the t2 equation.
            r_T2 += np.einsum('ijae,be->ijab', t2, F_MO[o:nbf,o:nbf])                                                                       # Contribution to Fae
            r_T2 -= np.einsum('imab,mj->ijab', t2, F_MO[0:o,0:o])                                                                           # Contribution to Fmi
            r_T2 += 0.5 * np.einsum('mnab,mnij->ijab', t2, ERI_MO[0:o,0:o,0:o,0:o])                                                         # Contribution to Tmnab and Wmnij
            r_T2 += 0.5 * np.einsum('ijef,abef->ijab', t2, ERI_MO[o:nbf,o:nbf,o:nbf,o:nbf])                                                 # Contribution to Tejif and Wabef
            r_T2 += np.einsum('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), ERI_MO[0:o,o:nbf,o:nbf,0:o])                                      # Contribution to Wmbej
            r_T2 += np.einsum('imae,mbej->ijab', t2, (ERI_MO[0:o,o:nbf,o:nbf,0:o] - ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o]))             # Contribution to Wmbej
            r_T2 -= np.einsum('mjae,mbie->ijab', t2, ERI_MO[0:o,o:nbf,0:o,o:nbf])                                                           # Contribution to Wmbej

            r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
            r_T2 -= E_CID * t2
            t2 += r_T2 / Dijab

            # Compute new CID energy.
            E_CID = np.einsum('ijab,ijab->', t2, 2 * ERI_MO[0:o,0:o,o:nbf,o:nbf] - ERI_MO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])

            # Compute total energy.
            E_tot = self.E_tot + E_CID

            # Compute convergence data.
            rms_t2 = np.einsum('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CID_old - E_CID

            #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CID.real, E_CID.imag, E_tot, delta_E.real, delta_E.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        return E_CID, t2



    def solve_CID_SO(self):
        """ 
        Solves the CID amplitudes and energies.
        """

        # For readability.
        nbf = 2*self.nbf
        o = 2*self.no
        v = 2*self.nv
        F_MO = self.F_MO.copy()
        ERI_MO = self.ERI_MO.copy()

        # Compute Fock matrix in spin-orbital basis.
        F_SO = compute_F_SO(self.wfn, F_MO)

        # Compute ERI in spin-orbital basis.
        ERI_SO = compute_ERI_SO(self.wfn, ERI_MO)
        
        # Swap axes for Dirac notation.
        ERI_SO = ERI_SO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the numerators for the T2 guess amplitudes.
        t2 = ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[0:o,0:o,o:nbf,o:nbf] - ERI_SO.copy().swapaxes(0,2).swapaxes(1,3).swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf]

        # Set up the denominators for the T2 guess amplitudes.
        Dijab = np.ones_like(ERI_SO)
        for i in range(0,o):
            for j in range(0,o):
                for a in range(o,nbf):
                    for b in range(o,nbf):
                        Dijab[i][j][a][b] *= F_SO[i][i] + F_SO[j][j] - F_SO[a][a] - F_SO[b][b]
        Dijab = Dijab[0:o,0:o,o:nbf,o:nbf]

        # Initial T2 guess amplitude.
        t2 = t2 / Dijab

        # Solve for initial CID energy.
        E_CID =  0.25 * np.einsum('ijab,ijab->', t2, ERI_SO[0:o,0:o,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])
        t2 = t2.copy()
        #print(E_CID)

        # Start iterative procedure.
        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_CID_old = E_CID
            t2_old = t2.copy()

            # Solving for the residual. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            r_T2 = ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[0:o,0:o,o:nbf,o:nbf] - ERI_SO.copy().swapaxes(0,2).swapaxes(1,3).swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf]        # First term of the t2 equation.
            r_T2 += np.einsum('ijae,be->ijab', t2, F_SO[o:nbf,o:nbf]) + np.einsum('ijeb,ae->ijab', t2, F_SO[o:nbf,o:nbf])                                               # Contribution to Fae
            r_T2 -= np.einsum('imab,mj->ijab', t2, F_SO[0:o,0:o]) + np.einsum('mjab,mi->ijab', t2, F_SO[0:o,0:o])                                                       # Contribution to Fmi
            r_T2 += 0.5 * np.einsum('mnab,mnij->ijab', t2, ERI_SO[0:o,0:o,0:o,0:o] - ERI_SO.swapaxes(2,3)[0:o,0:o,0:o,0:o])                                             # Contribution to Tmnab and Wmnij
            r_T2 += 0.5 * np.einsum('ijef,abef->ijab', t2, ERI_SO[o:nbf,o:nbf,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[o:nbf,o:nbf,o:nbf,o:nbf])                             # Contribution to Tejif and Wabef
            r_T2 += np.einsum('imae,mbej->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])                                           # Contribution to Wmbej
            r_T2 += np.einsum('mjae,mbei->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])                                           # Contribution to Wmbej
            r_T2 += np.einsum('imeb,maej->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])                                           # Contribution to Wmbej
            r_T2 += np.einsum('mjeb,maei->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])                                           # Contribution to Wmbej

            r_T2 -= E_CID * t2
            t2 += r_T2 / Dijab

            # Compute new CID energy.
            E_CID = 0.25 * np.einsum('ijab,ijab->', t2, ERI_SO[0:o,0:o,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])

            # Compute total energy.
            E_tot = self.E_tot + E_CID

            # Compute convergence data.
            rms_t2 = np.einsum('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CID_old - E_CID

            #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CID.real, E_CID.imag, E_tot, delta_E.real, delta_E.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        return E_CID, t2



    def solve_CISD_SO(self):
        """ 
        Solves the CISD amplitudes and energies.
        """

        # For readability.
        nbf = 2*self.nbf
        o = 2*self.no
        v = 2*self.nv
        F_MO = self.F_MO.copy()
        ERI_MO = self.ERI_MO.copy()

        # Compute Fock matrix in spin-orbital basis.
        F_SO = compute_F_SO(self.wfn, F_MO)

        # Compute ERI in spin-orbital basis.
        ERI_SO = compute_ERI_SO(self.wfn, ERI_MO)
    
        # Swap axes for Dirac notation.
        ERI_SO = ERI_SO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the numerator for the T1 guess amplitudes.
        t1 = np.zeros_like(F_SO[0:o,o:nbf])

        # Set up the denominator for the T1 guess amplitudes.
        Dia = np.ones_like(F_SO)
        for i in range(0,o):
            for a in range(o,nbf):
                Dia[i][a] *= F_SO[i][i] - F_SO[a][a]
        Dia = Dia[0:o,o:nbf]

        # Initial T1 guess amplitude.
        t1 = t1 / Dia

        # Set up the numerators for the T2 guess amplitudes.
        t2 = ERI_SO.copy()[0:o,0:o,o:nbf,o:nbf] - ERI_SO.copy().swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf]

        # Set up the denominators for the T2 guess amplitudes.
        Dijab = np.ones_like(ERI_SO)
        for i in range(0,o):
            for j in range(0,o):
                for a in range(o,nbf):
                    for b in range(o,nbf):
                        Dijab[i][j][a][b] *= F_SO[i][i] + F_SO[j][j] - F_SO[a][a] - F_SO[b][b]
        Dijab = Dijab[0:o,0:o,o:nbf,o:nbf]

        # Initial T2 guess amplitude.
        t2 = t2 / Dijab

        # Solve for initial CID energy.
        E_CISD =  np.einsum('ia,ia->', t1, F_SO[0:o,o:nbf] ) + 0.25 * np.einsum('ijab,ijab->', t2, ERI_SO[0:o,0:o,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])
        t1 = t1.copy()
        t2 = t2.copy()
        #print(E_CISD)

        # Start iterative procedure.
        iteration = 1 
        while iteration <= self.parameters['max_iterations']:
            E_CISD_old = E_CISD
            t1_old = t1.copy()
            t2_old = t2.copy()

            # Solving for the residuals. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            r_T1 = F_SO[0:o,o:nbf].copy()
            r_T1 -= np.einsum('ji,ja->ia', F_SO[0:o,0:o], t1)
            r_T1 += np.einsum('ab,ib->ia', F_SO[o:nbf,o:nbf], t1)
            r_T1 += np.einsum('jabi,jb->ia', ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t1)
            r_T1 += np.einsum('jb,ijab->ia', F_SO[0:o,o:nbf], t2)
            r_T1 += 0.5 * np.einsum('ajcb,ijcb->ia', ERI_SO[o:nbf,0:o,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[o:nbf,0:o,o:nbf,o:nbf], t2)
            r_T1 -= 0.5 * np.einsum('kjib,kjab->ia', ERI_SO[0:o,0:o,0:o,o:nbf] - ERI_SO.swapaxes(2,3)[0:o,0:o,0:o,o:nbf], t2)
            r_T1 -= E_CISD * t1

            r_T2 = ERI_SO[0:o,0:o,o:nbf,o:nbf].copy() - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf].copy()
            r_T2 -= np.einsum('kbij,ka->ijab', ERI_SO[0:o,o:nbf,0:o,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,0:o,0:o], t1)
            r_T2 -= np.einsum('akij,kb->ijab', ERI_SO[o:nbf,0:o,0:o,0:o] - ERI_SO.swapaxes(2,3)[o:nbf,0:o,0:o,0:o], t1)
            r_T2 += np.einsum('abcj,ic->ijab', ERI_SO[o:nbf,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[o:nbf,o:nbf,o:nbf,0:o], t1)
            r_T2 += np.einsum('abic,jc->ijab', ERI_SO[o:nbf,o:nbf,0:o,o:nbf] - ERI_SO.swapaxes(2,3)[o:nbf,o:nbf,0:o,o:nbf], t1)
            r_T2 += np.einsum('bc,ijac->ijab', F_SO[o:nbf,o:nbf], t2)
            r_T2 += np.einsum('ac,ijcb->ijab', F_SO[o:nbf,o:nbf], t2)
            r_T2 -= np.einsum('kj,ikab->ijab', F_SO[0:o,0:o], t2)
            r_T2 -= np.einsum('ki,kjab->ijab', F_SO[0:o,0:o], t2)
            r_T2 += 0.5 * np.einsum('klij,klab->ijab', ERI_SO[0:o,0:o,0:o,0:o] - ERI_SO.swapaxes(2,3)[0:o,0:o,0:o,0:o], t2)
            r_T2 += 0.5 * np.einsum('abcd,ijcd->ijab', ERI_SO[o:nbf,o:nbf,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[o:nbf,o:nbf,o:nbf,o:nbf], t2)
            r_T2 += np.einsum('kbcj,ikac->ijab', ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 += np.einsum('kbci,kjac->ijab', ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 += np.einsum('kacj,ikcb->ijab', ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 += np.einsum('kaci,kjcb->ijab', ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 -= E_CISD * t2

            t1 += r_T1 /Dia
            t2 += r_T2 / Dijab

            # Compute new CISD energy.
            E_CISD = np.einsum('ia,ia->', t1, F_SO[0:o,o:nbf]) + 0.25 * np.einsum('ijab,ijab->', t2, ERI_SO[0:o,0:o,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])

            # Compute total energy.
            E_tot = self.E_tot + E_CISD

            # Compute convergence data.
            rms_t1 = np.einsum('ia,ia->', t1_old - t1, t1_old - t1)
            rms_t1 = np.sqrt(rms_t1)

            rms_t2 = np.einsum('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CISD_old - E_CISD

            #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CISD.real, E_CISD.imag, E_tot, delta_E.real, delta_E.imag, rms_t1.real, rms_t1.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t1 < self.parameters['d_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        return E_CISD, t1, t2



    def solve_CISD(self):
        """ 
        Solves the CISD amplitudes and energies.
        """

        # For readability.
        nbf = self.nbf
        o = self.no
        v = self.nv
        F_MO = self.F_MO.copy()
        ERI_MO = self.ERI_MO.copy()

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the numerator for the T1 guess amplitudes.
        t1 = np.zeros_like(F_MO[0:o,o:nbf])

        # Set up the denominator for the T1 guess amplitudes.
        Dia = np.ones_like(F_MO)
        for i in range(0,o):
            for a in range(o,nbf):
                Dia[i][a] *= F_MO[i][i] - F_MO[a][a]
        Dia = Dia[0:o,o:nbf]

        # Initial T1 guess amplitude.
        t1 = t1 / Dia

        # Set up the numerators for the T2 guess amplitudes.
        t2 = ERI_MO.copy()[0:o,0:o,o:nbf,o:nbf] - ERI_MO.copy().swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf]

        # Set up the denominators for the T2 guess amplitudes.
        Dijab = np.ones_like(ERI_MO)
        for i in range(0,o):
            for j in range(0,o):
                for a in range(o,nbf):
                    for b in range(o,nbf):
                        Dijab[i][j][a][b] *= F_MO[i][i] + F_MO[j][j] - F_MO[a][a] - F_MO[b][b]
        Dijab = Dijab[0:o,0:o,o:nbf,o:nbf]

        # Initial T2 guess amplitude.
        t2 = t2 / Dijab

        # Solve for initial CID energy.
        E_CISD =  2.0 * np.einsum('ia,ia->', t1, F_MO[0:o,o:nbf]) + np.einsum('ijab,ijab->', t2, 2.0 * ERI_MO[0:o,0:o,o:nbf,o:nbf] - ERI_MO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])
        t1 = t1.copy()
        t2 = t2.copy()
        #print(E_CISD)

        # Start iterative procedure.
        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_CISD_old = E_CISD
            t1_old = t1.copy()
            t2_old = t2.copy()

            # Solving for the residuals. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            r_T1 = F_MO.copy().swapaxes(0,1)[0:o,o:nbf]
            r_T1 -= np.einsum('ji,ja->ia', F_MO[0:o,0:o], t1)
            r_T1 += np.einsum('ab,ib->ia', F_MO[o:nbf,o:nbf], t1)
            r_T1 += np.einsum('jabi,jb->ia', 2.0 * ERI_MO[0:o,o:nbf,o:nbf,0:o] - ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t1)
            r_T1 += np.einsum('jb,ijab->ia', F_MO[0:o,o:nbf], 2.0 * t2 - t2.swapaxes(2,3))
            r_T1 += np.einsum('ajbc,ijbc->ia', 2.0 * ERI_MO[o:nbf,0:o,o:nbf,o:nbf] - ERI_MO.swapaxes(2,3)[o:nbf,0:o,o:nbf,o:nbf], t2)
            r_T1 -= np.einsum('kjib,kjab->ia', 2.0 * ERI_MO[0:o,0:o,0:o,o:nbf] - ERI_MO.swapaxes(2,3)[0:o,0:o,0:o,o:nbf], t2)
            r_T1 -= E_CISD * t1

            r_T2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[0:o,0:o,o:nbf,o:nbf]
            r_T2 += np.einsum('abcj,ic->ijab', ERI_MO[o:nbf,o:nbf,o:nbf,0:o], t1)
            r_T2 += np.einsum('abic,jc->ijab', ERI_MO[o:nbf,o:nbf,0:o,o:nbf], t1)
            r_T2 -= np.einsum('kbij,ka->ijab', ERI_MO[0:o,o:nbf,0:o,0:o], t1)
            r_T2 -= np.einsum('akij,kb->ijab', ERI_MO[o:nbf,0:o,0:o,0:o], t1)
            r_T2 += np.einsum('ac,ijcb->ijab', F_MO[o:nbf,o:nbf], t2)
            r_T2 += np.einsum('bc,ijac->ijab', F_MO[o:nbf,o:nbf], t2)
            r_T2 -= np.einsum('ki,kjab->ijab', F_MO[0:o,0:o], t2)
            r_T2 -= np.einsum('kj,ikab->ijab', F_MO[0:o,0:o], t2)
            r_T2 += np.einsum('klij,klab->ijab', ERI_MO[0:o,0:o,0:o,0:o], t2)
            r_T2 += np.einsum('abcd,ijcd->ijab', ERI_MO[o:nbf,o:nbf,o:nbf,o:nbf], t2)            
            r_T2 -= np.einsum('kbcj,ikca->ijab', ERI_MO[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 += np.einsum('kaci,kjcb->ijab', 2.0 * ERI_MO[0:o,o:nbf,o:nbf,0:o] - ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 -= np.einsum('kbic,kjac->ijab', ERI_MO[0:o,o:nbf,0:o,o:nbf], t2)
            r_T2 -= np.einsum('kaci,kjbc->ijab', ERI_MO[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 += np.einsum('kbcj,ikac->ijab', 2.0 * ERI_MO[0:o,o:nbf,o:nbf,0:o] - ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            r_T2 -= np.einsum('kajc,ikcb->ijab', ERI_MO[0:o,o:nbf,0:o,o:nbf], t2)
            r_T2 -= E_CISD * t2

            t1 += r_T1 /Dia
            t2 += r_T2 / Dijab

            # Compute new CISD energy.
            E_CISD =  2.0 * np.einsum('ia,ia->', t1, F_MO[0:o,o:nbf]) + np.einsum('ijab,ijab->', t2, 2.0 * ERI_MO[0:o,0:o,o:nbf,o:nbf] - ERI_MO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])

            # Compute total energy.
            E_tot = self.E_tot + E_CISD

            # Compute convergence data.
            rms_t1 = np.einsum('ia,ia->', t1_old - t1, t1_old - t1)
            rms_t1 = np.sqrt(rms_t1)

            rms_t2 = np.einsum('ijab,ijab->', t2_old - t2, t2_old - t2)
            rms_t2 = np.sqrt(rms_t2)
            delta_E = E_CISD_old - E_CISD

            #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, E_CISD.real, E_CISD.imag, E_tot, delta_E.real, delta_E.imag, rms_t1.real, rms_t1.imag, rms_t2.real, rms_t2.imag))

            if iteration > 1:
                if abs(delta_E) < self.parameters['e_convergence'] and rms_t1 < self.parameters['d_convergence'] and rms_t2 < self.parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if iteration == self.parameters['max_iterations']:
                if abs(delta_E) > self.parameters['e_convergence'] or rms_t2 > self.parameters['d_convergence']:
                    print("Not converged.")
            iteration += 1

        return E_CISD, t1, t2

