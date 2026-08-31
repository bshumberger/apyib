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
from apyib.vg_apts_fd import VG_APT
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


def _aat_displacement_worker(parameters, ref_geom_str, unperturbed_C,
                              disp_type, disp_idx, disp_sign, h):
    """Run one displaced-geometry SCF for AAT finite-difference.

    Spawned workers receive only picklable objects.  The unperturbed BasisSet
    is rebuilt here from ``ref_geom_str`` (cheap — no SCF, just shell data).
    The displaced BasisSet is returned as a geometry string so the main process
    can reconstruct the Psi4 object after all workers finish.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    ref_geom_str : str
        Psi4 geometry string for the reference (unperturbed) geometry.
    unperturbed_C : ndarray
        MO coefficients at the reference geometry (for phase correction).
    disp_type : str
        ``'nuc'`` for a nuclear coordinate displacement, ``'mag'`` for a
        magnetic field perturbation.
    disp_idx : int
        Displacement index: 0…3N-1 for ``'nuc'``, 0…2 for ``'mag'``.
    disp_sign : int
        +1 or -1.
    h : float
        Perturbation step size.

    Returns
    -------
    C : ndarray
        Phase-corrected MO coefficients at the displaced geometry.
    T_list : list
        Amplitude list at the displaced geometry.
    displaced_geom_str : str
        Psi4 geometry string at the displaced geometry; used by the main
        process to reconstruct the displaced BasisSet object.
    """
    import psi4 as _psi4
    from apyib.energy import phase_corrected_energy as _pce

    params = copy.deepcopy(parameters)

    # Rebuild the unperturbed BasisSet from the reference geometry string.
    # BasisSet.build only constructs shell data — no SCF, no ERI.
    _psi4.core.clean_options()
    _psi4.set_options({'basis': params['basis']})
    _psi4.set_options({'freeze_core': params['freeze_core']})
    ref_mol = _psi4.geometry(ref_geom_str)
    unperturbed_basis = _psi4.core.BasisSet.build(ref_mol)

    # Apply the displacement.
    if disp_type == 'nuc':
        mol = _psi4.geometry(ref_geom_str)
        geom = mol.geometry().np.copy()
        geom[disp_idx // 3][disp_idx % 3] += disp_sign * h
        mol.set_geometry(_psi4.core.Matrix.from_array(geom))
        params['geom'] = mol.create_psi4_string_from_molecule()
    else:  # 'mag'
        params['F_mag'][disp_idx] += disp_sign * h

    _, T_list, C, _ = _pce(params, unperturbed_basis, unperturbed_C)
    return C, T_list, params['geom']


def _vg_apt_displacement_worker(parameters, ref_geom_str, unperturbed_C,
                                 disp_type, disp_idx, disp_sign, h):
    """Run one displaced-geometry SCF for VG APT finite-difference.

    Identical in structure to :func:`_aat_displacement_worker` but perturbs
    the vector potential field (``F_mom``) rather than the magnetic field
    (``F_mag``) for momentum-field displacements.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    ref_geom_str : str
        Psi4 geometry string for the reference (unperturbed) geometry.
    unperturbed_C : ndarray
        MO coefficients at the reference geometry (for phase correction).
    disp_type : str
        ``'nuc'`` for a nuclear coordinate displacement, ``'mom'`` for a
        vector potential field perturbation.
    disp_idx : int
        Displacement index: 0…3N-1 for ``'nuc'``, 0…2 for ``'mom'``.
    disp_sign : int
        +1 or -1.
    h : float
        Perturbation step size.

    Returns
    -------
    C : ndarray
        Phase-corrected MO coefficients at the displaced geometry.
    T_list : list
        Amplitude list at the displaced geometry.
    displaced_geom_str : str
        Psi4 geometry string at the displaced geometry; used by the main
        process to reconstruct the displaced BasisSet object.
    """
    import psi4 as _psi4
    from apyib.energy import phase_corrected_energy as _pce

    params = copy.deepcopy(parameters)

    _psi4.core.clean_options()
    _psi4.set_options({'basis': params['basis']})
    _psi4.set_options({'freeze_core': params['freeze_core']})
    ref_mol = _psi4.geometry(ref_geom_str)
    unperturbed_basis = _psi4.core.BasisSet.build(ref_mol)

    if disp_type == 'nuc':
        mol = _psi4.geometry(ref_geom_str)
        geom = mol.geometry().np.copy()
        geom[disp_idx // 3][disp_idx % 3] += disp_sign * h
        mol.set_geometry(_psi4.core.Matrix.from_array(geom))
        params['geom'] = mol.create_psi4_string_from_molecule()
    else:  # 'mom'
        params['F_mom'][disp_idx] += disp_sign * h

    _, T_list, C, _ = _pce(params, unperturbed_basis, unperturbed_C)
    return C, T_list, params['geom']


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


def compute_parallel_aats(parameters, nuc_pert_strength, mag_pert_strength,
                           normalization='full', num_processes=4):
    """Compute finite-difference AATs with all displaced-geometry SCFs parallelized.

    All 2*(3N) nuclear and 6 magnetic-field displacement SCFs are dispatched
    simultaneously to spawned workers; each worker rebuilds the unperturbed
    BasisSet internally (cheap — no SCF) and returns only picklable objects
    (numpy arrays and a geometry string).  The main process reconstructs the
    displaced BasisSet objects before building the AAT contraction object.
    The subsequent tensor contractions also run in a spawned pool.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    nuc_pert_strength : float
        Step size for nuclear coordinate displacements (Bohr).
    mag_pert_strength : float
        Step size for magnetic field perturbations (a.u.).
    normalization : str, optional
        Normalization convention passed to the AAT compute functions.
    num_processes : int, optional
        Number of worker processes for each multiprocessing pool (default 4).

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
    n = 3 * natom
    ref_geom_str = parameters['geom']

    # Enumerate all 2n + 6 displacement jobs.
    # Job order: nuc_pos[0..n-1], nuc_neg[0..n-1], mag_pos[0..2], mag_neg[0..2].
    jobs = (
        [(parameters, ref_geom_str, C, 'nuc', alpha, +1, nuc_pert_strength) for alpha in range(n)] +
        [(parameters, ref_geom_str, C, 'nuc', alpha, -1, nuc_pert_strength) for alpha in range(n)] +
        [(parameters, ref_geom_str, C, 'mag', beta,  +1, mag_pert_strength) for beta  in range(3)] +
        [(parameters, ref_geom_str, C, 'mag', beta,  -1, mag_pert_strength) for beta  in range(3)]
    )

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_psi4_worker_init) as pool:
        disp_results = pool.starmap(_aat_displacement_worker, jobs)

    # Unpack results by job-order slice.
    nuc_pos_C    = [disp_results[alpha][0]          for alpha in range(n)]
    nuc_pos_T    = [disp_results[alpha][1]          for alpha in range(n)]
    nuc_pos_geom = [disp_results[alpha][2]          for alpha in range(n)]
    nuc_neg_C    = [disp_results[n + alpha][0]      for alpha in range(n)]
    nuc_neg_T    = [disp_results[n + alpha][1]      for alpha in range(n)]
    nuc_neg_geom = [disp_results[n + alpha][2]      for alpha in range(n)]
    mag_pos_C    = [disp_results[2*n + beta][0]     for beta  in range(3)]
    mag_pos_T    = [disp_results[2*n + beta][1]     for beta  in range(3)]
    mag_neg_C    = [disp_results[2*n + 3 + beta][0] for beta  in range(3)]
    mag_neg_T    = [disp_results[2*n + 3 + beta][1] for beta  in range(3)]

    # Reconstruct displaced BasisSet objects from geometry strings.
    # BasisSet.build is cheap (no SCF); only shell data is constructed.
    psi4.core.clean_options()
    psi4.set_options({'basis': parameters['basis']})
    psi4.set_options({'freeze_core': parameters['freeze_core']})
    nuc_pos_basis = [psi4.core.BasisSet.build(psi4.geometry(g)) for g in nuc_pos_geom]
    nuc_neg_basis = [psi4.core.BasisSet.build(psi4.geometry(g)) for g in nuc_neg_geom]
    # Magnetic field displacements do not change the geometry, so the displaced
    # BasisSet is identical to the reference.
    mag_pos_basis = [basis] * 3
    mag_neg_basis = [basis] * 3

    # Build AAT contraction object (consumes BasisSets during init, stores only
    # numpy arrays on self — making the object picklable for the second pool).
    AATs = AAT(parameters, wfn, C, basis, T_list,
               nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
               mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T,
               nuc_pert_strength, mag_pert_strength)

    # Dispatch tensor contractions to a second spawned pool.
    lab = [[a, b, normalization] for a in range(3 * natom) for b in range(3)]

    method = parameters['method']
    with ctx.Pool(processes=num_processes, initializer=_psi4_worker_init) as pool:
        if method in _SPATIAL_METHODS:
            I = pool.starmap(AATs.compute_spatial_aats, lab)
        elif method in _SO_METHODS:
            I = pool.starmap(AATs.compute_SO_aats, lab)
        else:
            raise ValueError(
                f"Unknown method '{method}' for parallel AAT computation. "
                f"Spatial: {sorted(_SPATIAL_METHODS)}, SO: {sorted(_SO_METHODS)}"
            )

    I = np.reshape(np.array(I), (3 * natom, 3))
    print(I, "\n")
    return I


def compute_parallel_vg_apt(parameters, nuc_pert_strength, mom_pert_strength,
                              normalization='full', num_processes=4):
    """Compute finite-difference VG APTs with all displaced-geometry SCFs parallelized.

    All 2*(3N) nuclear and 6 vector-potential displacement SCFs are dispatched
    simultaneously to spawned workers; each worker rebuilds the unperturbed
    BasisSet internally (cheap — no SCF) and returns only picklable objects
    (numpy arrays and a geometry string).  The main process reconstructs the
    displaced BasisSet objects before building the VG APT contraction object.
    The subsequent tensor contractions also run in a spawned pool.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.  Must include ``'F_mom': [0.0, 0.0, 0.0]`` or
        it will be inserted with that default before dispatching workers.
    nuc_pert_strength : float
        Step size for nuclear coordinate displacements (Bohr).
    mom_pert_strength : float
        Step size for vector potential (momentum) field perturbations (a.u.).
    normalization : str, optional
        Normalization convention passed to the VG APT compute functions.
    num_processes : int, optional
        Number of worker processes for each multiprocessing pool (default 4).

    Returns
    -------
    P : ndarray, shape (3*natom, 3)
        Velocity-gauge atomic polar tensor array.
    """
    parameters.setdefault('F_mom', [0.0, 0.0, 0.0])

    E_list, T_list, C, basis = energy(parameters)
    print("Total Energy: ", total_energy(E_list))

    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)
    natom = H.molecule.natom()
    n = 3 * natom
    ref_geom_str = parameters['geom']

    # Enumerate all 2n + 6 displacement jobs.
    # Job order: nuc_pos[0..n-1], nuc_neg[0..n-1], mom_pos[0..2], mom_neg[0..2].
    jobs = (
        [(parameters, ref_geom_str, C, 'nuc', alpha, +1, nuc_pert_strength) for alpha in range(n)] +
        [(parameters, ref_geom_str, C, 'nuc', alpha, -1, nuc_pert_strength) for alpha in range(n)] +
        [(parameters, ref_geom_str, C, 'mom', beta,  +1, mom_pert_strength) for beta  in range(3)] +
        [(parameters, ref_geom_str, C, 'mom', beta,  -1, mom_pert_strength) for beta  in range(3)]
    )

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_psi4_worker_init) as pool:
        disp_results = pool.starmap(_vg_apt_displacement_worker, jobs)

    # Unpack results by job-order slice.
    nuc_pos_C    = [disp_results[alpha][0]          for alpha in range(n)]
    nuc_pos_T    = [disp_results[alpha][1]          for alpha in range(n)]
    nuc_pos_geom = [disp_results[alpha][2]          for alpha in range(n)]
    nuc_neg_C    = [disp_results[n + alpha][0]      for alpha in range(n)]
    nuc_neg_T    = [disp_results[n + alpha][1]      for alpha in range(n)]
    nuc_neg_geom = [disp_results[n + alpha][2]      for alpha in range(n)]
    mom_pos_C    = [disp_results[2*n + beta][0]     for beta  in range(3)]
    mom_pos_T    = [disp_results[2*n + beta][1]     for beta  in range(3)]
    mom_neg_C    = [disp_results[2*n + 3 + beta][0] for beta  in range(3)]
    mom_neg_T    = [disp_results[2*n + 3 + beta][1] for beta  in range(3)]

    # Reconstruct displaced BasisSet objects from geometry strings.
    psi4.core.clean_options()
    psi4.set_options({'basis': parameters['basis']})
    psi4.set_options({'freeze_core': parameters['freeze_core']})
    nuc_pos_basis = [psi4.core.BasisSet.build(psi4.geometry(g)) for g in nuc_pos_geom]
    nuc_neg_basis = [psi4.core.BasisSet.build(psi4.geometry(g)) for g in nuc_neg_geom]
    # Momentum field displacements do not change the geometry.
    mom_pos_basis = [basis] * 3
    mom_neg_basis = [basis] * 3

    # Build VG APT contraction object (stores only numpy arrays on self).
    VG_APTs = VG_APT(parameters, wfn, C, basis, T_list,
                     nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
                     mom_pos_C, mom_neg_C, mom_pos_basis, mom_neg_basis, mom_pos_T, mom_neg_T,
                     nuc_pert_strength, mom_pert_strength)

    # Dispatch tensor contractions to a second spawned pool.
    lab = [[a, b, normalization] for a in range(3 * natom) for b in range(3)]

    method = parameters['method']
    with ctx.Pool(processes=num_processes, initializer=_psi4_worker_init) as pool:
        if method in _SPATIAL_METHODS:
            P = pool.starmap(VG_APTs.compute_spatial_vg_apts, lab)
        elif method in _SO_METHODS:
            P = pool.starmap(VG_APTs.compute_SO_vg_apts, lab)
        else:
            raise ValueError(
                f"Unknown method '{method}' for parallel VG APT computation. "
                f"Spatial: {sorted(_SPATIAL_METHODS)}, SO: {sorted(_SO_METHODS)}"
            )

    P = np.reshape(np.array(P), (3 * natom, 3))
    print(P, "\n")
    return P
