""" Contains energy function call. If implementing new electronic structure methods, the method needs to be included here. """

import psi4
import numpy as np
import scipy.linalg as la
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO
from apyib.utils import compute_mo_overlap
from apyib.utils import compute_so_overlap
from apyib.utils import compute_phase
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn

def energy(parameters, print_level=0):
    # Set the Hamiltonian and perform a standard Hartree-Fock calculation.
    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)
    E_SCF, E_tot, C = wfn.solve_SCF(parameters)

    # Obtaining basis and nuclear repulsion energy.
    basis = H.basis_set
    E_nuc = H.E_nuc

    # Set the number of atoms.
    natom = H.molecule.natom()

    # Setting up return values to be overwritten.
    E = 0
    t0 = 1
    t1 = 0
    t2 = 0

    # Perform calculations for the chosen method if not Hartree-Fock. The wavefunction is returned in the spatial-orbital basis.
    if parameters['method'] == 'MP2':
        wfn_MP2 = mp2_wfn(parameters, E_SCF, E_tot, C)
        E, t2 = wfn_MP2.solve_MP2()
    if parameters['method'] == 'CID':
        wfn_cid = ci_wfn(parameters, E_SCF, E_tot, C)
        E, t2 = wfn_cid.solve_CID()

    # Perform calculations for the chosen method if not Hartree-Fock. The wavefunction is returned in the spin-orbital basis.
    if parameters['method'] == 'MP2_SO':
        wfn_MP2 = mp2_wfn(parameters, E_SCF, E_tot, C)
        E, t2 = wfn_MP2.solve_MP2_SO()
    if parameters['method'] == 'CID_SO':
        wfn_cid = ci_wfn(parameters, E_SCF, E_tot, C)
        E, t2 = wfn_cid.solve_CID_SO()
    if parameters['method'] == 'CISD_SO':
        wfn_cisd = ci_wfn(parameters, E_SCF, E_tot, C)
        E, t1, t2 = wfn_cisd.solve_CISD_SO()

    # Setting up return lists.
    E_list = [E_SCF, E, E_nuc]
    T_list = [t0, t1, t2]

    # Setting up print options.
    if print_level > 0:
        print("Method: ", parameters['method'])
        print("Electronic Hartree-Fock Energy: ", E_SCF)
        if parameters['method'] != 'RHF':
            print("Electronic Post-Hartree-Fock Energy: ", E)
        print("Total Energy: ", E_tot + E)

    return E_list, T_list, C, basis



def phase_corrected_energy(parameters, unperturbed_basis, unperturbed_C, print_level=0):
    # Set the Hamiltonian and perform a standard Hartree-Fock calculation.
    H = Hamiltonian(parameters)
    wfn = hf_wfn(H)
    E_SCF, E_tot, C = wfn.solve_SCF(parameters)

    # Obtaining basis and nuclear repulsion energy.
    basis = H.basis_set
    E_nuc = H.E_nuc

    # Set the number of atoms.
    natom = H.molecule.natom()

    # Setting up return values to be overwritten.
    E = 0 
    t0 = 1 
    t1 = 0 
    t2 = 0 

    # Correct the phase.
    phase_corrected_C = compute_phase(wfn.ndocc, wfn.nbf, unperturbed_basis, unperturbed_C, basis, C)

    # Perform calculations for the chosen method if not Hartree-Fock. The wavefunction is returned in the spatial-orbital basis.
    if parameters['method'] == 'MP2':
        wfn_MP2 = mp2_wfn(parameters, E_SCF, E_tot, phase_corrected_C)
        E, t2 = wfn_MP2.solve_MP2()
    if parameters['method'] == 'CID':
        wfn_cid = ci_wfn(parameters, E_SCF, E_tot, phase_corrected_C)
        E, t2 = wfn_cid.solve_CID()

    # Perform calculations for the chosen method if not Hartree-Fock. The wavefunction is returned in the spin-orbital basis.
    if parameters['method'] == 'MP2_SO':
        wfn_MP2 = mp2_wfn(parameters, E_SCF, E_tot, phase_corrected_C)
        E, t2 = wfn_MP2.solve_MP2_SO()
    if parameters['method'] == 'CID_SO':
        wfn_cid = ci_wfn(parameters, E_SCF, E_tot, phase_corrected_C)
        E, t2 = wfn_cid.solve_CID_SO()
    if parameters['method'] == 'CISD_SO':
        wfn_cisd = ci_wfn(parameters, E_SCF, E_tot, phase_corrected_C)
        E, t1, t2 = wfn_cisd.solve_CISD_SO()

    # Setting up return lists.
    E_list = [E_SCF, E, E_nuc]
    T_list = [t0, t1, t2] 

    # Setting up print options.
    if print_level > 0:
        print("Method: ", parameters['method'])
        print("Electronic Hartree-Fock Energy: ", E_SCF)
        if parameters['method'] != 'RHF':
            print("Electronic Post-Hartree-Fock Energy: ", E)
        print("Total Energy: ", E_tot + E)

    return E_list, T_list, phase_corrected_C, basis



