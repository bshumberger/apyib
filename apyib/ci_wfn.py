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
        t2 = ERI_MO.copy()[0:o,0:o,o:nbf,o:nbf]

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

            # Trying with pycc type algorithm.
            r_T2 = 0.5 * ERI_MO[0:o,0:o,o:nbf,o:nbf].copy()                                                                                 # First term of the t2 equation.
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
        E_CID =  0.25 * np.einsum('ijab,ijab->', t2, ERI_SO[0:o,0:o,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])
        t2 = t2.copy()
        #print(E_CID)

        # Start iterative procedure.
        iteration = 1
        while iteration <= self.parameters['max_iterations']:
            E_CID_old = E_CID
            t2_old = t2.copy()

            # Solving for the residual. Note that the equations for the T2 amplitudes must all be permuted to have final indices of i,j,a,b for Tijab.
            r_T2 = ERI_SO[0:o,0:o,o:nbf,o:nbf].copy() - ERI_SO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf].copy()                                    # First term of the t2 equation.
            r_T2 += np.einsum('ijae,be->ijab', t2, F_SO[o:nbf,o:nbf]) + np.einsum('ijeb,ae->ijab', t2, F_SO[o:nbf,o:nbf])                   # Contribution to Fae
            r_T2 -= np.einsum('imab,mj->ijab', t2, F_SO[0:o,0:o]) + np.einsum('mjab,mi->ijab', t2, F_SO[0:o,0:o])                           # Contribution to Fmi
            r_T2 += 0.5 * np.einsum('mnab,mnij->ijab', t2, ERI_SO[0:o,0:o,0:o,0:o] - ERI_SO.swapaxes(2,3)[0:o,0:o,0:o,0:o])                 # Contribution to Tmnab and Wmnij
            r_T2 += 0.5 * np.einsum('ijef,abef->ijab', t2, ERI_SO[o:nbf,o:nbf,o:nbf,o:nbf] - ERI_SO.swapaxes(2,3)[o:nbf,o:nbf,o:nbf,o:nbf]) # Contribution to Tejif and Wabef
            r_T2 += np.einsum('imae,mbej->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])               # Contribution to Wmbej
            r_T2 += np.einsum('mjae,mbei->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])               # Contribution to Wmbej
            r_T2 += np.einsum('imeb,maej->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])               # Contribution to Wmbej
            r_T2 += np.einsum('mjeb,maei->ijab', t2, ERI_SO[0:o,o:nbf,o:nbf,0:o] - ERI_SO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o])               # Contribution to Wmbej

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





    

