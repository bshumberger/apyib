"""Contains functions for parallel finite-difference property computations."""

import os
import copy
import numpy as np
import psi4
import multiprocessing as mp
from apyib.hamiltonian import Hamiltonian
from apyib.energy import energy
from apyib.hf_wfn import hf_wfn
from apyib.fin_diff import finite_difference
from apyib.aats import AAT
from apyib.utils import total_energy

_SPATIAL_METHODS = frozenset({'RHF', 'MP2', 'CID', 'CISD'})
_SO_METHODS = frozenset({'MP2_SO', 'CID_SO', 'CISD_SO'})


# ---------------------------------------------------------------------------
# Worker helpers — must be module-level so multiprocessing can pickle them.
#
# Design: each worker receives a deep copy of 'parameters' (a plain dict of
# scalars/lists — cheap to pickle) plus a small numpy array (geom_ref).
# The worker applies its specific displacement, calls energy(), and returns
# only a scalar.  No Psi4 objects cross the process boundary.
#
# _psi4_worker_init() runs once per spawned process before any job starts;
# it redirects Psi4 output to /dev/null so workers don't write to output.dat.
# ---------------------------------------------------------------------------

def _psi4_worker_init():
    """Redirect Psi4 output to /dev/null in each worker process."""
    import psi4 as _psi4
    _psi4.core.set_output_file(os.devnull, False)


def _hessian_worker(parameters, geom_ref, sa, alpha, sb, beta, h):
    """Return E(alpha displaced by sa*h, beta displaced by sb*h).

    sa, sb ∈ {+1, -1}.  alpha//3 is the atom index, alpha%3 is x/y/z.
    """
    import psi4 as _psi4
    from apyib.energy import energy as _energy
    from apyib.utils import total_energy as _te

    params = copy.deepcopy(parameters)
    mol = _psi4.geometry(params['geom'])
    geom = geom_ref.copy()
    geom[alpha // 3][alpha % 3] += sa * h
    geom[beta // 3][beta % 3] += sb * h
    mol.set_geometry(_psi4.core.Matrix.from_array(geom))
    params['geom'] = mol.create_psi4_string_from_molecule()
    E_list, _, _, _ = _energy(params)
    return _te(E_list)


def _apt_worker(parameters, geom_ref, sa, alpha, sb_field, beta_field, h_nuc, h_elec):
    """Return E(nuc_alpha displaced by sa*h_nuc, F_el[beta_field] += sb_field*h_elec).

    sa, sb_field ∈ {+1, -1}.  beta_field ∈ {0,1,2} for x/y/z field component.
    """
    import psi4 as _psi4
    from apyib.energy import energy as _energy
    from apyib.utils import total_energy as _te

    params = copy.deepcopy(parameters)
    mol = _psi4.geometry(params['geom'])
    geom = geom_ref.copy()
    geom[alpha // 3][alpha % 3] += sa * h_nuc
    mol.set_geometry(_psi4.core.Matrix.from_array(geom))
    params['geom'] = mol.create_psi4_string_from_molecule()
    params['F_el'][beta_field] += sb_field * h_elec
    E_list, _, _, _ = _energy(params)
    return _te(E_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_parallel_hessian(parameters, nuc_pert_strength, num_processes=4):
    """Compute the nuclear Hessian by central FD using a multiprocessing pool.

    All 4*(3N)^2 displaced-geometry energy evaluations run in parallel.
    Each worker runs an independent Psi4 SCF (and post-HF if requested).

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    nuc_pert_strength : float
        Step size for nuclear coordinate displacements (Bohr).
    num_processes : int, optional
        Number of worker processes (default 4).

    Returns
    -------
    hessian : ndarray, shape (3*natom, 3*natom)
        Second derivative of total energy w.r.t. nuclear coordinates [E_h / a_0^2].
    """
    mol = psi4.geometry(parameters['geom'])
    natom = mol.natom()
    n = 3 * natom
    h = nuc_pert_strength
    geom_ref = mol.geometry().np.copy()

    # Enumerate all 4*(3N)^2 independent doubly-displaced-geometry energy
    # evaluations.  Each job is (sa, alpha, sb, beta) where sa/sb ∈ {+1,-1}.
    # List order: sa outer → alpha → sb → beta, giving flat index
    #   i = sa_idx*(2*n^2) + alpha*(2*n) + sb_idx*n + beta
    # which maps cleanly to reshape(2, n, 2, n) → E[sa_idx, alpha, sb_idx, beta].
    jobs = [(parameters, geom_ref, sa, alpha, sb, beta, h)
            for sa in (+1, -1) for alpha in range(n)
            for sb in (+1, -1) for beta in range(n)]

    # mp.get_context('spawn') starts fresh Python interpreters so Psi4 global
    # state is not shared; _psi4_worker_init suppresses file-based output.
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_psi4_worker_init) as pool:
        results = pool.starmap(_hessian_worker, jobs)

    # Reconstruct the four doubly-displaced energy tables and form the Hessian
    # by the same double central-difference formula as the serial version:
    #   H[a,b] = (E(+a,+b) - E(+a,-b) - E(-a,+b) + E(-a,-b)) / 4h^2
    E = np.array(results).reshape(2, n, 2, n)  # E[sa_idx, alpha, sb_idx, beta]
    return (E[0, :, 0, :] - E[0, :, 1, :] - E[1, :, 0, :] + E[1, :, 1, :]) / (4 * h**2)


def compute_parallel_apt(parameters, nuc_pert_strength, elec_pert_strength, num_processes=4):
    """Compute the atomic polar tensor by central FD using a multiprocessing pool.

    All 4*(3N)*3 displaced energy evaluations run in parallel.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    nuc_pert_strength : float
        Step size for nuclear coordinate displacements (Bohr).
    elec_pert_strength : float
        Step size for electric field perturbations (a.u.).
    num_processes : int, optional
        Number of worker processes (default 4).

    Returns
    -------
    P : ndarray, shape (3*natom, 3)
        Atomic polar tensor.
    """
    mol = psi4.geometry(parameters['geom'])
    natom = mol.natom()
    n = 3 * natom
    h_nuc = nuc_pert_strength
    h_elec = elec_pert_strength
    geom_ref = mol.geometry().np.copy()

    # Enumerate all 4*(3N)*3 independent (nuc, field) displaced-energy jobs.
    # List order: sa → alpha → sb_field → beta_field
    # → reshape(2, n, 2, 3) → E[sa_idx, alpha, sb_field_idx, beta_field]
    jobs = [(parameters, geom_ref, sa, alpha, sb_field, beta_field, h_nuc, h_elec)
            for sa in (+1, -1) for alpha in range(n)
            for sb_field in (+1, -1) for beta_field in range(3)]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_psi4_worker_init) as pool:
        results = pool.starmap(_apt_worker, jobs)

    # mu_beta = -dE/dF_beta; APT[alpha, beta] = dmu_beta/dR_alpha
    # P[a,b] = -(E(+a,+F_b) - E(+a,-F_b) - E(-a,+F_b) + E(-a,-F_b)) / 4*h_nuc*h_elec
    E = np.array(results).reshape(2, n, 2, 3)  # E[sa_idx, alpha, sb_field_idx, beta_field]
    pos_mu = -(E[0, :, 0, :] - E[0, :, 1, :]) / (2 * h_elec)
    neg_mu = -(E[1, :, 0, :] - E[1, :, 1, :]) / (2 * h_elec)
    return (pos_mu - neg_mu) / (2 * h_nuc)


def compute_parallel_aats(parameters, nuc_pert_strength, mag_pert_strength, normalization='full', num_processes=4):
    """Compute finite-difference atomic axial tensors in parallel.

    The FD displaced-geometry SCF calculations run serially (Psi4 BasisSet
    objects cannot cross process boundaries); the subsequent AAT tensor
    contractions are distributed across worker processes.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    nuc_pert_strength : float
        Step size for nuclear coordinate displacements.
    mag_pert_strength : float
        Step size for magnetic field perturbations.
    normalization : str, optional
        Normalization convention passed to the AAT compute functions.
    num_processes : int, optional
        Number of worker processes for the multiprocessing pool.

    Returns
    -------
    I : ndarray, shape (3*natom, 3)
        Atomic axial tensor array.
    """
    E_list, T_list, C, basis = energy(parameters)
    print("Total Energy: ", total_energy(E_list))

    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)
    natom = H.molecule.natom()

    aat_finite_difference = finite_difference(parameters, basis, C)
    (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis,
     nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C,
     mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T) = aat_finite_difference.compute_AAT(
         nuc_pert_strength, mag_pert_strength)

    AATs = AAT(parameters, wfn, C, basis, T_list,
               nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
               mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T,
               nuc_pert_strength, mag_pert_strength)

    lab = [[a, b, normalization] for a in range(3 * natom) for b in range(3)]

    method = parameters['method']
    pool = mp.Pool(processes=num_processes)
    if method in _SPATIAL_METHODS:
        I_async = pool.starmap_async(AATs.compute_spatial_aats, lab)
    elif method in _SO_METHODS:
        I_async = pool.starmap_async(AATs.compute_SO_aats, lab)
    else:
        pool.close()
        pool.join()
        raise ValueError(
            f"Unknown method '{method}' for parallel AAT computation. "
            f"Spatial: {sorted(_SPATIAL_METHODS)}, SO: {sorted(_SO_METHODS)}"
        )

    pool.close()
    pool.join()
    I = np.reshape(np.array(I_async.get()), (3 * natom, 3))
    print(I, "\n")

    return I
