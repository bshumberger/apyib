"""Contains the second-order Moller-Plesset perturbation theory (MP2) wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import solve_DIIS

class mp2_wfn(object):
    """ 
    Wavefunction object.
    """
    # Define the specific properties of the MP2 wavefunction.
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
        self.nv = self.nbf - self.no
 
    # Compute the MP2 wavefunction and energy.
    def solve_MP2(self):

        # Transform the AO ERIs to the MO basis.
        #ERI_MO = np.zeros_like(self.H.ERI)
        #for p in range(nbf):
        #    for q in range(nbf):
        #        for r in range(nbf):
        #            for s in range(nbf):
        #                for mu in range(nbf):
        #                    for nu in range(nbf):
        #                        for lambd in range(nbf):
        #                            for sigma in range(nbf):
        #                                ERI_MO[p,q,r,s] += C[mu,p] * C[nu,q] * ERI[mu,nu,lambd,sigma] * C[lambd,r] * C[sigma,s]

        #ERI_MO = np.einsum('mp,nq,mnlg,lr,gs->pqrs', np.conjugate(self.C), np.conjugate(self.C), self.H.ERI, self.C, self.C)

        #ERI_MO = np.einsum('mnlg,gs->mnls', self.H.ERI, self.C)
        #ERI_MO = np.einsum('mnls,lr->mnrs', ERI_MO, np.conjugate(self.C))
        #ERI_MO = np.einsum('nq,mnrs->mqrs', self.C, ERI_MO)
        #ERI_MO = np.einsum('mp,mqrs->pqrs', np.conjugate(self.C), ERI_MO)

        # Compute the MP2 energy.
        #E_MP2 = 0 
        #for i in range(self.no):
        #    for j in range(self.no):
        #        for a in range(self.no,self.nbf):
        #            for b in range(self.no,self.nbf):
        #                E_MP2 += ( ERI_MO[i,a,j,b] * ( 2 * ERI_MO[i,a,j,b] - ERI_MO[i,b,j,a] ) ) / ( self.e[i] + self.e[j] - self.e[a] - self.e[b] )

        #E_tot = self.E_tot + E_MP2

        # Transform the AO ERIs to the MO basis.
        ERI_MO = np.einsum('mnlg,gs->mnls', self.H.ERI, self.C)
        ERI_MO = np.einsum('mnls,lr->mnrs', ERI_MO, np.conjugate(self.C))
        ERI_MO = np.einsum('nq,mnrs->mqrs', self.C, ERI_MO)
        ERI_MO = np.einsum('mp,mqrs->pqrs', np.conjugate(self.C), ERI_MO)

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the denominators for the T2 guess amplitudes. Note that this is an equivalent formulation as above.
        Dijab = np.ones_like(ERI_MO)
        for i in range(0,self.no):
            for j in range(0,self.no):
                for a in range(self.no,self.nbf):
                    for b in range(self.no,self.nbf):
                        Dijab[i][j][a][b] *= self.e[i] + self.e[j] - self.e[a] - self.e[b]
        Dijab = Dijab[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf]

        # Initial T2 guess amplitude.
        t2 = ERI_MO[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf] / Dijab

        # Compute the MP2 energy.
        E_MP2 = np.einsum('ijab,ijab->', 2 * ERI_MO[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf] - ERI_MO.swapaxes(2,3)[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf], t2)


        return E_MP2, t2




