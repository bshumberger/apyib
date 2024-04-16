"""Contains the second-order Moller-Plesset perturbation theory (MP2) wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import solve_DIIS
from apyib.utils import compute_F_MO
from apyib.utils import compute_ERI_MO
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO

class mp2_wfn(object):
    """ 
    Wavefunction object.
    """
    # Define the specific properties of the MP2 wavefunction.
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
        self.nv = self.nbf - self.no
 
    # Compute the MP2 wavefunction and energy.
    def solve_MP2(self):
        # Compute MO Fock matrix.
        F_MO = compute_F_MO(self.parameters, self.H, self.wfn, self.C)

        # Compute MO electron repulsion integrals.
        ERI_MO = compute_ERI_MO(self.parameters, self.H, self.wfn, self.C)

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the denominators for the T2 guess amplitudes. Note that this is an equivalent formulation as above.
        Dijab = np.ones_like(ERI_MO)
        for i in range(0,self.no):
            for j in range(0,self.no):
                for a in range(self.no,self.nbf):
                    for b in range(self.no,self.nbf):
                        Dijab[i][j][a][b] *= F_MO[i][i] + F_MO[j][j] - F_MO[a][a] - F_MO[b][b]
        Dijab = Dijab[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf]

        # Initial T2 guess amplitude.
        t2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf] / Dijab

        # Compute the MP2 energy.
        E_MP2 = np.einsum('ijab,ijab->', 2 * ERI_MO[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf] - ERI_MO.swapaxes(2,3)[0:self.no,0:self.no,self.no:self.nbf,self.no:self.nbf], t2)

        return E_MP2, t2



    # Compute the MP2 wavefunction and energy from spin-orbital expressions.
    def solve_MP2_SO(self):
        # Compute Fock matrix in spin-orbital basis.
        F_MO = compute_F_MO(self.parameters, self.H, self.wfn, self.C)
        F_SO = compute_F_SO(self.wfn, F_MO)

        # Compute ERI in spin-orbital basis.
        ERI_MO = compute_ERI_MO(self.parameters, self.H, self.wfn, self.C)
        ERI_SO = compute_ERI_SO(self.wfn, ERI_MO)

        # Swap axes for Dirac notation.
        ERI_SO = ERI_SO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Set up the denominators for the T2 guess amplitudes. Note that this is an equivalent formulation as above.
        Dijab = np.ones_like(ERI_SO)
        for i in range(0,2*self.no):
            for j in range(0,2*self.no):
                for a in range(2*self.no,2*self.nbf):
                    for b in range(2*self.no,2*self.nbf):
                        Dijab[i][j][a][b] *= (F_SO[i][i] + F_SO[j][j] - F_SO[a][a] - F_SO[b][b])
        Dijab = Dijab[0:2*self.no,0:2*self.no,2*self.no:2*self.nbf,2*self.no:2*self.nbf]

        # Initial T2 guess amplitude.
        t2 = (ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[0:2*self.no,0:2*self.no,2*self.no:2*self.nbf,2*self.no:2*self.nbf] - ERI_SO.copy().swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)[0:2*self.no,0:2*self.no,2*self.no:2*self.nbf,2*self.no:2*self.nbf]) / Dijab

        # Compute the MP2 energy.
        E_MP2 = 0.25 * np.einsum('ijab,ijab->', ERI_SO[0:2*self.no,0:2*self.no,2*self.no:2*self.nbf,2*self.no:2*self.nbf] - ERI_SO.swapaxes(2,3)[0:2*self.no,0:2*self.no,2*self.no:2*self.nbf,2*self.no:2*self.nbf], t2) 

        return E_MP2, t2






