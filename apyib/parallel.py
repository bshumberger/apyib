"""Contains the functions associated with computing the AATs in parallel."""

import numpy as np
import multiprocessing as mp
from apyib.hamiltonian import Hamiltonian
from apyib.energy import energy
from apyib.hf_wfn import hf_wfn
from apyib.fin_diff import finite_difference
from apyib.aats import AAT
from apyib.utils import total_energy

_SPATIAL_METHODS = frozenset({'RHF', 'MP2', 'CID', 'CISD'})
_SO_METHODS = frozenset({'MP2_SO', 'CID_SO', 'CISD_SO'})


def compute_parallel_aats(parameters, nuc_pert_strength, mag_pert_strength, normalization='full', num_processes=4):
    """Compute finite-difference atomic axial tensors in parallel.

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
