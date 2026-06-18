"""Base class for analytic derivative calculations."""

import numpy as np
import psi4
import opt_einsum as oe
import multiprocessing as mp
from types import SimpleNamespace
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.utils import get_slices


def _cphf_fill_U(U_ov_k, A, F, D_oo, D_vv,
                 nbf, no, nv, o, v, f_, o_, v_,
                 ov_sign, S_k, h_dep_k, dep_sign, do_dep, orbitals, dtype):
    """Fill one CPHF U matrix from its pre-solved OV block.

    Module-level so it is picklable for multiprocessing.Pool.

    Parameters
    ----------
    U_ov_k : ndarray, shape (nv, no)
        Solved OV block for perturbation k, from G @ B_all[:,k].
    A : ndarray, shape (nbf, nbf, nbf, nbf)
        Full CPHF A-tensor.
    F : ndarray, shape (nbf, nbf)
        Reference Fock matrix.
    D_oo, D_vv : ndarray
        Orbital-energy denominators for occupied-occupied and
        virtual-virtual dependent-pair equations.
    nbf, no, nv : int
        Orbital counts.
    o, v : slice
        Occupied and virtual slices.
    f_, o_, v_ : slice
        Frozen-core, active occupied, and virtual slices for block-
        diagonal constraints.
    ov_sign : {-1, +1}
        -1: U[o,v] = -U[v,o].T (real); +1: U[o,v] = +U[v,o].T (imag).
    S_k : ndarray or None
        Full overlap-derivative matrix for nuclear perturbations; None
        for field perturbations.
    h_dep_k : ndarray or None
        Perturbation operator for dependent-pair B-vectors; None to skip.
    dep_sign : {+1, -1}
        Sign of h in dependent-pair B vector (+1 real, -1 magnetic).
    do_dep : bool
        Whether to solve dependent-pair equations.
    orbitals : str
        'non-canonical' or 'canonical'.
    dtype : type
        numpy dtype for U (float or complex).

    Returns
    -------
    U : ndarray, shape (nbf, nbf)
    """
    U = np.zeros((nbf, nbf), dtype=dtype)
    U[v, o] = U_ov_k

    # OV/VO symmetry: -S[o,v] term is zero for field perturbations.
    U[o, v] = ov_sign * U[v, o].T - (S_k[o, v] if S_k is not None else 0)

    if do_dep:
        h = h_dep_k
        S = S_k
        B_oo = dep_sign * h[o, o] + oe.contract('em,iejm->ij', U[v, o], A.swapaxes(1, 2)[o, v, o, o])
        B_vv = dep_sign * h[v, v] + oe.contract('em,aebm->ab', U[v, o], A.swapaxes(1, 2)[v, v, v, o])
        if S is not None:
            # Nuclear-only S-overlap corrections to dependent-pair B vectors.
            B_oo -= oe.contract('ij,jj->ij', S[o, o], F[o, o])
            B_oo -= 0.5 * oe.contract('mn,imjn->ij', S[o, o], A.swapaxes(1, 2)[o, o, o, o])
            B_vv -= oe.contract('ab,bb->ab', S[v, v], F[v, v])
            B_vv -= 0.5 * oe.contract('mn,ambn->ab', S[o, o], A.swapaxes(1, 2)[v, o, v, o])
        U[o, o] = B_oo / D_oo
        U[v, v] = B_vv / D_vv

        # Diagonal override from orthonormality: -0.5*S for nuclear, 0 for fields.
        if S is not None:
            for j in range(no):
                U[j, j] = -0.5 * S[j, j]
            for b in range(no, nbf):
                U[b, b] = -0.5 * S[b, b]
        else:
            for j in range(no):
                U[j, j] = 0.0
            for b in range(no, nbf):
                U[b, b] = 0.0

    # Block-diagonal constraints for non-canonical (block-diagonal) MOs.
    if orbitals == 'non-canonical':
        if S_k is not None:
            U[f_, f_] = -0.5 * S_k[f_, f_]
            U[o_, o_] = -0.5 * S_k[o_, o_]
            U[v_, v_] = -0.5 * S_k[v_, v_]
        else:
            U[f_, f_] = 0.0
            U[o_, o_] = 0.0
            U[v_, v_] = 0.0

    return U


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
        """Run a single RHF SCF and store the converged wavefunction.

        Parameters
        ----------
        parameters : dict
            Calculation parameters (see :func:`apyib.energy.energy`).
        """
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

    def _solve_cphf(self, G, A, B_all, nbf, no, nv, o, v, F, orbitals, f_, o_, v_,
                    ov_sign=-1, S_all=None, h_dep_all=None, dep_sign=+1,
                    dtype=float, num_processes=1):
        """Solve the CPHF equations for npert perturbation directions at once.

        The dominant step is the batched OV solve ``U_ov_all = G @ B_all``,
        a single BLAS dgemm call for all npert right-hand sides.
        Per-perturbation U-matrix filling can be run in parallel when
        ``num_processes > 1``.

        Parameters
        ----------
        G : ndarray, shape (nv*no, nv*no)
            Inverted CPHF kernel from :meth:`_build_cphf_A`.
        A : ndarray, shape (nbf, nbf, nbf, nbf)
            Full CPHF A-tensor for dependent-pair equations.
        B_all : ndarray, shape (nv*no, npert)
            RHS vectors; one column per perturbation direction.
            Nuclear: ``-F_a[v,o] + S-overlap terms``.
            Electric (LG): ``-h[v,o]``.
            Magnetic / velocity gauge: ``+h[v,o]``.
        nbf, no, nv : int
            Basis function, occupied, and virtual orbital counts.
        o, v : slice
            Occupied and virtual orbital index slices.
        F : ndarray, shape (nbf, nbf)
            Reference Fock matrix in the MO basis.
        orbitals : {'non-canonical', 'canonical'}
            Controls block-diagonal constraint filling.
        f_, o_, v_ : slice
            Frozen-core, active occupied, and virtual index slices.
        ov_sign : {-1, +1}
            Sign in ``U[o,v] = ov_sign * U[v,o].T - S[o,v]``.
            Use ``-1`` for real perturbations (nuclear, electric length gauge);
            use ``+1`` for imaginary perturbations (magnetic, velocity gauge).
        S_all : ndarray, shape (npert, nbf, nbf) or None
            Full overlap-derivative matrices for nuclear perturbations.
            Provides ``S[o,v]``, ``S[o,o]``, ``S[v,v]`` for U filling.
            Pass ``None`` for field perturbations where overlap derivatives vanish.
        h_dep_all : ndarray, shape (npert, nbf, nbf) or None
            Perturbation operator for dependent-pair B-vectors.
            Nuclear: perturbed Fock ``F_a``.
            Electric (LG): dipole ``mu``.
            Magnetic / velocity gauge: ``L/2`` or nabla.
            Pass ``None`` to skip dependent-pair equations.
        dep_sign : {+1, -1}
            Sign of h in the dependent-pair B vector.
            Use ``+1`` for real perturbations, ``-1`` for magnetic /
            velocity gauge.
        dtype : type
            numpy dtype for U matrices; ``float`` for real perturbations,
            ``complex`` for imaginary perturbations.
        num_processes : int
            Number of parallel worker processes for the per-perturbation U
            filling.  Default 1 runs sequentially; ``> 1`` spawns a
            ``multiprocessing.Pool``.

        Returns
        -------
        U_list : list of ndarray, each shape (nbf, nbf)
            CPHF U-coefficient matrices, one per perturbation direction.

        Notes
        -----
        Phase-space CPHF (future): the phase-space approach uses a
        momentum-weighted CPHF kernel ``G_ps`` and a differently constructed
        ``B_all``.  Pass those objects here; the solve structure is identical.
        """
        npert = B_all.shape[1]

        # Vectorized OV solve: single batched dgemm for all npert directions.
        U_ov_all = (G @ B_all).reshape(nv, no, npert)

        freeze_core = self.parameters.get('freeze_core', False)
        do_dep = (h_dep_all is not None and
                  (freeze_core is True or orbitals == 'canonical'))

        D_oo = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1, 1)) + np.eye(no)
        D_vv = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1, 1)) + np.eye(nv)

        args_list = [
            (U_ov_all[:, :, k],
             A, F, D_oo, D_vv,
             nbf, no, nv, o, v, f_, o_, v_,
             ov_sign,
             S_all[k] if S_all is not None else None,
             h_dep_all[k] if h_dep_all is not None else None,
             dep_sign, do_dep, orbitals, dtype)
            for k in range(npert)
        ]

        if num_processes > 1:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_processes) as pool:
                U_list = pool.starmap(_cphf_fill_U, args_list)
        else:
            U_list = [_cphf_fill_U(*args) for args in args_list]

        return U_list

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
