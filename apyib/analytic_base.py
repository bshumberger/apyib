"""Base class for analytic derivative calculations."""

import numpy as np
import psi4
import opt_einsum as oe
from types import SimpleNamespace
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import get_slices


class AnalyticDerivative:
    """Base class for analytic derivative objects (Hessian, APTs, AATs).

    Performs a single RHF SCF on construction. Subclasses inherit the
    converged wavefunction and provide property-specific compute methods.

    Parameters
    ----------
    parameters : dict
        Calculation parameters (see :func:`apyib.energy.energy`).
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.H = Hamiltonian(parameters)
        self.wfn = hf_wfn(self.H)
        E_SCF, self.C = self.wfn.solve_SCF(parameters)

    def _setup_mo_basis(self):
        """Build the MO-basis quantities shared by all compute methods.

        Returns
        -------
        types.SimpleNamespace
            Attributes: ``C``, ``nbf``, ``no``, ``nv``, ``C_list``,
            ``I_list``, ``f_``, ``o_``, ``v_``, ``t_``, ``o``, ``v``,
            ``t``, ``C_p4``, ``natom``, ``atoms``, ``h``, ``ERI``, ``F``,
            ``mints``.
        """
        C = self.C
        nbf = self.wfn.nbf
        no = self.wfn.ndocc
        nv = nbf - no

        C_list, I_list = get_slices(self.parameters, self.wfn)
        f_ = C_list[0]
        o_ = C_list[1]
        v_ = C_list[2]
        t_ = C_list[3]

        o = slice(0, no)
        v = slice(no, nbf)
        t = slice(0, nbf)

        C_p4 = psi4.core.Matrix.from_array(C)
        natom = self.H.molecule.natom()
        atoms = np.arange(0, natom)

        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)

        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)
        ERI = ERI.swapaxes(1, 2)  # (pr|qs) -> <pq|rs>

        F = h + oe.contract('piqi->pq', 2 * ERI[:, o, :, o] - ERI.swapaxes(2, 3)[:, o, :, o])

        mints = psi4.core.MintsHelper(self.H.basis_set)

        return SimpleNamespace(
            C=C, nbf=nbf, no=no, nv=nv,
            C_list=C_list, I_list=I_list,
            f_=f_, o_=o_, v_=v_, t_=t_,
            o=o, v=v, t=t,
            C_p4=C_p4, natom=natom, atoms=atoms,
            h=h, ERI=ERI, F=F, mints=mints,
        )

    def _build_cphf_A(self, ERI, F, no, nv, o, v, sign=1):
        """Build the CPHF A matrix and its inverted ov-block kernel G.

        Parameters
        ----------
        ERI : ndarray
            Two-electron repulsion integrals in the MO basis (Dirac notation).
        F : ndarray
            Fock matrix in the MO basis.
        no : int
            Number of occupied orbitals.
        nv : int
            Number of virtual orbitals.
        o : slice
            Occupied orbital slice.
        v : slice
            Virtual orbital slice.
        sign : {+1, -1}, optional
            Sign of the exchange-like term in A.  Use ``+1`` (default) for
            real perturbations (nuclear displacement, length-gauge electric
            field) and ``-1`` for imaginary perturbations (magnetic field,
            velocity-gauge electric field).

        Returns
        -------
        A : ndarray, shape (nbf, nbf, nbf, nbf)
            The full CPHF A matrix.
        G : ndarray, shape (nv*no, nv*no)
            Inverted ov-block kernel used to solve the independent-pair
            CPHF equations.
        """
        J_K = 2 * ERI - ERI.swapaxes(2, 3)
        A = sign * J_K + J_K.swapaxes(1, 3)
        A = A.swapaxes(1, 2)
        G = (oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no),
                         F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no))
             + A[v, o, v, o])
        G = np.linalg.inv(G.reshape((nv * no, nv * no)))
        return A, G
