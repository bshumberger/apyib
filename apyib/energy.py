"""Contains energy function calls for all supported electronic structure methods."""

import psi4
from apyib.config import _VALID_METHODS
from apyib.utils import compute_phase, total_energy
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn


def _run_post_hf(parameters, wfn, print_level=0):
    """Dispatch to the post-HF solver selected by ``parameters['method']``.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    wfn : hf_wfn
        Converged RHF wavefunction.
    print_level : int, optional
        Verbosity level passed to iterative solvers.

    Returns
    -------
    E : float
        Post-HF correlation energy (0 for RHF).
    t1 : ndarray or int
        Singles amplitudes (0 for methods without singles).
    t2 : ndarray or int
        Doubles amplitudes (0 for RHF).
    """
    E = 0
    t1 = 0
    t2 = 0
    method = parameters['method']
    if method == 'RHF':
        pass
    elif method == 'MP2':
        wfn_mp2 = mp2_wfn(parameters, wfn)
        E, t2 = wfn_mp2.solve_MP2()
    elif method == 'CID':
        wfn_cid = ci_wfn(parameters, wfn)
        E, t2 = wfn_cid.solve_CID(print_level)
    elif method == 'CISD':
        wfn_cisd = ci_wfn(parameters, wfn)
        E, t1, t2 = wfn_cisd.solve_CISD(print_level)
    elif method == 'MP2_SO':
        wfn_mp2 = mp2_wfn(parameters, wfn)
        E, t2 = wfn_mp2.solve_MP2_SO()
    elif method == 'CID_SO':
        wfn_cid = ci_wfn(parameters, wfn)
        E, t2 = wfn_cid.solve_CID_SO(print_level)
    elif method == 'CISD_SO':
        wfn_cisd = ci_wfn(parameters, wfn)
        E, t1, t2 = wfn_cisd.solve_CISD_SO(print_level)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid methods: {_VALID_METHODS}"
        )
    return E, t1, t2


def energy(parameters, return_H=False, print_level=0):
    """Run an SCF + optional post-HF calculation.

    Parameters
    ----------
    parameters : dict
        Calculation parameters including ``'method'``.
    return_H : bool, optional
        If True, also return the :class:`~apyib.hamiltonian.Hamiltonian`.
    print_level : int, optional
        0 = silent, 1 = iteration table, 2 = full summary.

    Returns
    -------
    E_list : list
        ``[E_SCF, E_corr, E_nuc]``, plus ``T_nuc`` for phase-space runs.
    T_list : list
        ``[t0, t1, t2]`` amplitude data.
    C : ndarray
        MO coefficient matrix.
    basis : psi4 BasisSet
        AO basis set object.
    H : Hamiltonian
        Only returned when ``return_H=True``.
    """
    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)
    E_SCF, C = wfn.solve_SCF(parameters, print_level)

    basis = H.basis_set
    E_nuc = H.E_nuc

    E, t1, t2 = _run_post_hf(parameters, wfn, print_level)

    E_list = [E_SCF, E, E_nuc]
    T_list = [1, t1, t2]

    if parameters.get('P_nuc') is not None:
        E_list = [E_SCF, E, E_nuc, H.T_nuc]

    if print_level == 2:
        print("Method: ", parameters['method'])
        print("Electronic Hartree-Fock Energy: ", E_SCF)
        if parameters['method'] != 'RHF':
            print("Electronic Post-Hartree-Fock Energy: ", E)
        print("Total Energy: ", total_energy(E_list))
        print(parameters)

    if return_H:
        return E_list, T_list, C, basis, H
    return E_list, T_list, C, basis


def phase_corrected_energy(parameters, unperturbed_basis, unperturbed_C, return_H=False, print_level=0):
    """Run SCF + optional post-HF with MO phase correction for finite-difference use.

    Parameters
    ----------
    parameters : dict
        Calculation parameters.
    unperturbed_basis : psi4 BasisSet
        Basis set from the unperturbed reference calculation.
    unperturbed_C : ndarray
        MO coefficients from the unperturbed reference calculation.
    return_H : bool, optional
        If True, also return the :class:`~apyib.hamiltonian.Hamiltonian`.
    print_level : int, optional
        0 = silent, >0 = print energy summary.

    Returns
    -------
    E_list, T_list, C, basis[, H]
        Same as :func:`energy`.
    """
    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)
    E_SCF, C = wfn.solve_SCF(parameters, print_level)

    basis = H.basis_set
    E_nuc = H.E_nuc

    wfn.C = compute_phase(wfn.ndocc, wfn.nbf, unperturbed_basis, unperturbed_C, basis, C)

    E, t1, t2 = _run_post_hf(parameters, wfn, print_level)

    E_list = [E_SCF, E, E_nuc]
    T_list = [1, t1, t2]

    if parameters.get('P_nuc') is not None:
        E_list = [E_SCF, E, E_nuc, H.T_nuc]

    if print_level > 0:
        print("Method: ", parameters['method'])
        print("Electronic Hartree-Fock Energy: ", E_SCF)
        if parameters['method'] != 'RHF':
            print("Electronic Post-Hartree-Fock Energy: ", E)
        print("Total Energy: ", total_energy(E_list))

    if return_H:
        return E_list, T_list, wfn.C, basis, H
    return E_list, T_list, wfn.C, basis
