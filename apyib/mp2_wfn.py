"""Contains the second-order Moller-Plesset perturbation theory (MP2) wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
import opt_einsum as oe
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import solve_DIIS
from apyib.utils import get_slices
from apyib.utils import compute_F_MO
from apyib.utils import compute_ERI_MO
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO

class mp2_wfn(object):
    """ 
    Wavefunction object.
    """
    # Define the specific properties of the MP2 wavefunction.
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
        self.D_ijab = self.eps_o.reshape(-1,1,1,1) + self.eps_o.reshape(-1,1,1) - self.eps_v.reshape(-1,1) - self.eps_v

    # Compute the MP2 wavefunction and energy.
    def solve_MP2(self):
        # Setting up slice options for the MO integrals.
        o_ = self.I_list[1]
        v_ = self.I_list[2]

        # Compute MO electron repulsion integrals.
        ERI_MO = compute_ERI_MO(self.parameters, self.wfn, self.C_list)

        # Swap axes for Dirac notation.
        ERI_MO = ERI_MO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Initial T2 guess amplitude.
        t2 = ERI_MO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] / self.D_ijab

        # Compute the MP2 energy.
        E_MP2 = oe.contract('ijab,ijab->', 2 * ERI_MO[o_,o_,v_,v_] - ERI_MO.swapaxes(2,3)[o_,o_,v_,v_], t2)

        return E_MP2, t2



    # Compute the MP2 wavefunction and energy from spin-orbital expressions.
    def solve_MP2_SO(self):
        # Build energy denominators in the spin orbital basis.
        eps_o = np.repeat(self.eps_o, 2)
        eps_v = np.repeat(self.eps_v, 2)
        D_ijab = eps_o.reshape(-1,1,1,1) + eps_o.reshape(-1,1,1) - eps_v.reshape(-1,1) - eps_v

        # Setting up slice options for the MO integrals.
        o_ = self.I_list[1]
        v_ = self.I_list[2]

        # Compute ERI in spin-orbital basis.
        ERI_MO = compute_ERI_MO(self.parameters, self.wfn, self.C_list)
        ERI_SO = compute_ERI_SO(self.wfn, ERI_MO)

        # Swap axes for Dirac notation.
        ERI_SO = ERI_SO.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Initial T2 guess amplitude.
        t2 = (ERI_SO.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_] - ERI_SO.copy().swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]) / D_ijab

        # Compute the MP2 energy.
        E_MP2 = 0.25 * oe.contract('ijab,ijab->', ERI_SO[o_,o_,v_,v_] - ERI_SO.swapaxes(2,3)[o_,o_,v_,v_], t2) 

        return E_MP2, t2






