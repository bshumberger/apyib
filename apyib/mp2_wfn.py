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
    def __init__(self, parameters):

        # Define the Hamiltonian and the Hartree-Fock reference energy and wavefunction.
        self.parameters = parameters
        self.H = Hamiltonian(parameters)
        self.wfn = hf_wfn(self.H)
        self.e, self.E_SCF, self.E_tot, self.C = self.wfn.solve_SCF(parameters)

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

        self.H.ERI = np.einsum('mnlg,gs->mnls', self.H.ERI, self.C)
        self.H.ERI = np.einsum('mnls,lr->mnrs', self.H.ERI, self.C)
        self.H.ERI = np.einsum('nq,mnrs->mqrs', np.conjugate(self.C), self.H.ERI)
        ERI_MO = np.einsum('mp,mqrs->pqrs', np.conjugate(self.C), self.H.ERI)

        # Compute the MP2 energy.
        E_MP2 = 0 
        for i in range(self.no):
            for j in range(self.no):
                for a in range(self.no,self.nbf):
                    for b in range(self.no,self.nbf):
                        E_MP2 += ( ERI_MO[i,a,j,b] * ( 2 * ERI_MO[i,a,j,b] - ERI_MO[i,b,j,a] ) ) / ( self.e[i] + self.e[j] - self.e[a] - self.e[b] )

        E_tot = self.E_tot + E_MP2

        return E_tot, self.E_SCF, E_MP2, ERI_MO




