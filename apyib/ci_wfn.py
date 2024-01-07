"""Contains the configuration interaction doubles (CID) wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import solve_DIIS
from apyib.mp2_wfn import mp2_wfn

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)
class ci_wfn(object):
    """
    Wavefunction object.
    """
    # Define the specific properties of the CI wavefunction.
    def __init__(self, parameters, e, E_SCF, E_tot, C):

        # Define the Hamiltonian and the Hartree-Fock reference energy and wavefunction.
        self.parameters = parameters
        self.H = Hamiltonian(parameters)
        self.wfn = hf_wfn(self.H)
        #self.e, self.E_SCF, self.E_tot, self.C = self.wfn.solve_SCF(parameters)
        self.e = e
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
        self.F_MO, self.ERI_MO = self.AO_to_MO()
        self.ERI_MO = self.ERI_MO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the numerators for the T2 guess amplitudes.
        self.t2 = self.ERI_MO.copy()[0:o,0:o,o:nbf,o:nbf]

        # Set up the denominators for the T2 guess amplitudes.
        self.Dijab = np.ones_like(self.ERI_MO)
        for i in range(0,o):
            for j in range(0,o):
                for a in range(o,nbf):
                    for b in range(o,nbf):
                        self.Dijab[i][j][a][b] *= self.e[i] + self.e[j] - self.e[a] - self.e[b]
        self.Dijab = self.Dijab[0:o,0:o,o:nbf,o:nbf]

        # Initial T2 guess amplitude.
        self.t2 = self.t2 / self.Dijab

    # Convert AO Hamiltonian one- and two-electron integrals to MO one- and two-electron integrals.
    def AO_to_MO(self):
        
        # Set up the one-electron AO integrals.
        h_AO = self.H.T + self.H.V
        h_AO = h_AO.astype('complex128')
        for alpha in range(3):
            h_AO -=  self.parameters['F_el'][alpha] * self.H.mu_el[alpha] + self.parameters['F_mag'][alpha] * self.H.mu_mag[alpha]

        # Set up the two-electron AO integrals.
        ERI_AO = self.H.ERI.astype('complex128')

        # Set up Fock matrix elements.
        # Compute the density.
        D = np.zeros_like(h_AO)
        for mu in range(self.nbf):
            for nu in range(self.nbf):
                for m in range(self.no):
                    D[mu, nu] += self.C[mu, m] * np.conjugate(np.transpose(self.C[nu, m]))

        # Compute the Fock matrix elements.
        F_AO = np.zeros_like(h_AO)
        for mu in range(self.nbf):
            for nu in range(self.nbf):
                F_AO[mu, nu] += h_AO[mu, nu] 
                for lambd in range(self.nbf):
                    for sigma in range(self.nbf):
                        F_AO[mu, nu] += D[lambd, sigma] * ( 2 * ERI_AO[mu, nu, lambd, sigma] - ERI_AO[mu, lambd, nu, sigma] )

        # Compute MO Fock matrix elements.
        F_MO = np.einsum('ip,ij,jq->pq', np.conjugate(self.C),F_AO,self.C)

        # Compute the two-electron MO integrals.
        ERI_MO = np.einsum('mnlg,gs->mnls', self.H.ERI, self.C)
        ERI_MO = np.einsum('mnls,lr->mnrs', ERI_MO, np.conjugate(self.C))
        ERI_MO = np.einsum('nq,mnrs->mqrs', self.C, ERI_MO)
        ERI_MO = np.einsum('mp,mqrs->pqrs', np.conjugate(self.C), ERI_MO)

        return F_MO, ERI_MO

    def solve_CID(self):
        """
        Solves the CID amplitudes and energies.
        """

        # For readability.
        nbf = self.nbf
        o = self.no
        v = self.nv

        # Solve for initial CID energy.
        
        E_CID =  np.einsum('ijab,ijab->', self.t2, 2*self.ERI_MO[0:o,0:o,o:nbf,o:nbf]-self.ERI_MO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])
        t2 = self.t2.copy()
        #print(E_CID)

        # Start iterative procedure.
        iteration = 1 
        while iteration <= self.parameters['max_iterations']:
            E_CID_old = E_CID
            t2_old = t2.copy()

            # Compute new t2 amplitudes. The terms included come from the spin-integrated equations for t_{i_alpha j_beta}^{a_alpha b_beta}.
            #term1 = self.ERI_MO[0:o,0:o,o:nbf,o:nbf].copy()
            #term2 = np.einsum('bc,ijac->ijab', self.F_MO[o:nbf,o:nbf], t2) + np.einsum('ac,ijcb->ijab', self.F_MO[o:nbf,o:nbf], t2)
            #term3 = -np.einsum('kj,ikab->ijab', self.F_MO[0:o,0:o], t2) - np.einsum('ki,kjab->ijab', self.F_MO[0:o,0:o], t2)
            #term4 = 0.5 * np.einsum('klij,klab->ijab', self.ERI_MO[0:o,0:o,0:o,0:o], t2) + 0.5 * np.einsum('klji,lkab->jiab', self.ERI_MO.swapaxes(2,3)[0:o,0:o,0:o,0:o], t2)
            #term5 = 0.5 * np.einsum('abcd,ijcd->ijab', self.ERI_MO[o:nbf,o:nbf,o:nbf,o:nbf], t2) + 0.5 * np.einsum('abdc,ijdc->ijba', self.ERI_MO.swapaxes(2,3)[o:nbf,o:nbf,o:nbf,o:nbf], t2)
            #term6a = np.einsum('kbcj,ikac->ijab', self.ERI_MO[0:o,o:nbf,o:nbf,0:o], t2-t2.swapaxes(2,3)) + np.einsum('kaci,jkbc->jiba',self.ERI_MO[0:o,o:nbf,o:nbf,0:o], t2-t2.swapaxes(2,3)).swapaxes(0,1).swapaxes(2,3)
            #term6b = np.einsum('kbic,jkca->jiba', self.ERI_MO[0:o,o:nbf,0:o,o:nbf], t2) + np.einsum('kajc,ikcb->ijab',self.ERI_MO[0:o,o:nbf,0:o,o:nbf], t2)
            #term6c = np.einsum('kaci,jkbc->jiba', self.ERI_MO[0:o,o:nbf,o:nbf,0:o] - self.ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            #term6d = np.einsum('kbcj,ikac->ijab', self.ERI_MO[0:o,o:nbf,o:nbf,0:o] - self.ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o], t2)
            #r_t2 = term1 + term2 + term3 + term4 + term5 + term6a #+ term6b #+ term6c + term6d

            # Trying with pycc type algorithm.
            r_T2 = 0.5 * self.ERI_MO[0:o,0:o,o:nbf,o:nbf].copy()                                                                                # First term of the t2 equation.
            r_T2 += np.einsum('ijae,be->ijab', t2, self.F_MO[o:nbf,o:nbf])                                                                      # Contribution to Fae
            r_T2 -= np.einsum('imab,mj->ijab', t2, self.F_MO[0:o,0:o])                                                                          # Contribution to Fmi
            r_T2 += 0.5 * np.einsum('mnab,mnij->ijab', t2, self.ERI_MO[0:o,0:o,0:o,0:o])                                                        # Contribution to Tmnab and Wmnij
            r_T2 += 0.5 * np.einsum('ijef,abef->ijab', t2, self.ERI_MO[o:nbf,o:nbf,o:nbf,o:nbf])                                                # Contribution to Tejif and Wabef
            r_T2 += np.einsum('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), self.ERI_MO[0:o,o:nbf,o:nbf,0:o])                                     # Contribution to Wmbej
            r_T2 += np.einsum('imae,mbej->ijab', t2, (self.ERI_MO[0:o,o:nbf,o:nbf,0:o] - self.ERI_MO.swapaxes(2,3)[0:o,o:nbf,o:nbf,0:o]))       # Contribution to Wmbej
            r_T2 -= np.einsum('mjae,mbie->ijab', t2, self.ERI_MO[0:o,o:nbf,0:o,o:nbf])                                                          # Contribution to Wmbej

            r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
            r_T2 -= E_CID * t2
            t2 += r_T2 / self.Dijab

            #print(r_t2 - r_T2)
            # Compute new CID energy.
            E_CID = np.einsum('ijab,ijab->', t2, 2*self.ERI_MO[0:o,0:o,o:nbf,o:nbf] - self.ERI_MO.swapaxes(2,3)[0:o,0:o,o:nbf,o:nbf])

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




    

