"""This script contains a set of functions for comparisons, improved performance, and regular operations associated with multiple methods."""

import numpy as np
import psi4



def run_psi4(parameters, method='RHF'):
    """ 
    Run Psi4 for comparison to the apyib code. Note that psi4 does not do magnetic field perturbations.
    """
    # Set Psi4 options..
    psi4.core.clean_options()
    psi4.set_output_file("output.dat", False)
    psi4.set_memory('2 GB')

    # Input geometry for Psi4.
    molecule = psi4.geometry(parameters['geom'])
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    # Modify the Psi4 input to deal with electric field perturbations rather than scalar perturbations.
    F_el = [0.0, 0.0, 0.0]
    for axis in range(3):
        F_el[axis] += parameters['F_el'][axis] * -1

    # Set the basis set, convergence criteria, and perturbation for the calculation.
    psi4.set_options({'basis': parameters['basis'],
                    'scf_type':'pk',
                    'e_convergence': parameters['e_convergence'],
                    'd_convergence': parameters['d_convergence'],
                    'DIIS': parameters['DIIS'],
                    'freeze_core': parameters['freeze_core'],
                    'mp2_type': 'conv',
                    'PERTURB_H': True,
                    'PERTURB_WITH':'DIPOLE',
                    'PERTURB_DIPOLE': F_el})

    if method == 'RHF':
        # Run Psi4 Hartree-Fock code and return the energy and wavefunction.
        e, wfn = psi4.energy("scf", return_wfn=True)

    elif method == 'MP2':
        # Run Psi4 MP2 code and return the energy and wavefunction.
        e, wfn = psi4.energy("mp2", return_wfn=True)

    elif method == 'CISD':
        # Run Psi4 CISD code and return the energy and wavefunction.
        e, wfn = psi4.energy("cisd", return_wfn=True)

    return e, wfn



def solve_DIIS(parameters, F, D, S, X, F_iter, e_iter, min_DIIS=1, max_DIIS=7):
    """
    Solve the direct inversion of the iterative subspace (DIIS) equations for improved SCF convergence.
    """
    # Truncate the storage of the Fock matrices and error matrices for stability reasons.
    if len(e_iter) > max_DIIS:
        while len(e_iter) > max_DIIS:
            del F_iter[0]
            del e_iter[0]
    
    # Store Fock matrices.
    F_iter.append(F)
    
    # Compute the error matrix.
    e_diis = np.zeros_like(F)
    e_diis = X@(S@D@F - np.conjugate(np.transpose(S@D@F)))@X
    e_iter.append(e_diis)
    
    # Compute the "B" matrix.
    B = np.zeros((len(e_iter)+1,len(e_iter)+1))
    B[-1,:] = -1
    B[:,-1] = -1
    B[-1,-1] = 0 
    for m in range(len(e_iter)):
        for n in range(len(e_iter)):
            #B[m,n] = np.einsum('mn, mn->',e_iter[m], e_iter[n], optimize=True)
            for o in range(len(e_diis[0])):
                for p in range(len(e_diis[0])):
                    B[m,n] = B[m,n] + e_iter[m][o][p] * e_iter[n][o][p]
    
    # Build the "A" matrix for the system of linear equations.
    A = np.zeros(len(e_iter)+1)
    A[-1] = -1
    
    # Solve the system of linear equations.
    #C_diis = np.linalg.inv(B) @ A
    C_diis = np.linalg.solve(B,A)
    
    # Solve for new Fock matrix.
    F = np.zeros_like(F)
    for j in range(len(C_diis) - 1): 
        F = F + C_diis[j] * F_iter[j]

    return F



def get_slices(parameters, wfn):
    # Define the number of occupied and virtual orbitals.
    nfzc = wfn.H.basis_set.n_frozen_core()
    nbf = wfn.nbf
    no = wfn.ndocc
    nv = wfn.nbf - wfn.ndocc

    # Define the slices for molecular orbital coefficients (to be used in computing of F_MO and ERI_MO).
    f = slice(0, nfzc)
    o = slice(nfzc, no)
    v = slice(no, nbf)
    t = slice(nfzc, nbf)

    C_list = [f, o, v, t]

    # Define the slices for the molecular orbital basis integrals (to be used in computing E and the t-amplitudes).
    if parameters['method'] == 'RHF' or parameters['method'] == 'MP2' or parameters['method'] == 'CID' or parameters['method'] == 'CISD':
        f_ = slice(0, nfzc)
        o_ = slice(0, no - nfzc)
        v_ = slice(no - nfzc, nbf - nfzc)
        t_ = slice(0, nbf - nfzc)
    elif parameters['method'] == 'MP2_SO' or parameters['method'] == 'CID_SO' or parameters['method'] == 'CISD_SO':
        f_ = slice(0, 2*nfzc)
        o_ = slice(0, 2*no - 2*nfzc)
        v_ = slice(2*no - 2*nfzc, 2*nbf - 2*nfzc)
        t_ = slice(0, 2*nbf - 2*nfzc)

    I_list = [f_, o_, v_, t_]

    return C_list, I_list



def compute_F_MO(parameters, wfn, C_list): 
    """ 
    Computes the MO basis Fock matrix from the AO basis.
    """
    # Set up the slices for frozen core, total, occupied, and virtual orbital subspaces.
    f = C_list[0]
    o = C_list[1]
    v = C_list[2]
    t = C_list[3]

    # Get wavefunction coefficients.
    C = wfn.C

    # Set up the one-electron AO integrals.
    h_AO = wfn.H.T + wfn.H.V

    # Set up the two-electron AO integrals.
    ERI_AO = wfn.H.ERI.copy()

    # Compute frozen core energy and append frozen core operator.
    E_fc = 0
    if parameters['freeze_core'] == True:
        C_fc = C[:,f]
        D_fc = np.einsum('mp,np->mn', C_fc, np.conjugate(C_fc))
        h_AO_fc = h_AO + np.einsum('ls,mnls->mn', D_fc, 2 * ERI_AO - ERI_AO.swapaxes(1,2))
        E_fc = np.einsum('nm,mn->', D_fc, h_AO + h_AO_fc)
        h_AO = h_AO_fc

    # Compute the density.
    D = np.einsum('mp,np->mn', C[:,o], np.conjugate(C[:,o]))

    # Compute the Fock matrix elements.
    F_AO = h_AO + np.einsum('ls,mnls->mn', D, 2 * ERI_AO - ERI_AO.swapaxes(1,2))

    # Compute MO Fock matrix elements.
    F_MO = np.einsum('ip,ij,jq->pq', np.conjugate(C[:,t]), F_AO, C[:,t])

    return F_MO, E_fc



def compute_ERI_MO(parameters, wfn, C_list):
    """
    Computes the MO basis electron repulsion integrals from the AO basis.
    """
    # Set up the slices for frozen core, total, occupied, and virtual orbital subspaces.
    f = C_list[0]
    o = C_list[1]
    v = C_list[2]
    t = C_list[3]

    C = wfn.C.copy()[:,t]

    # Set up the two-electron AO integrals.
    ERI_AO = wfn.H.ERI.copy()

    # Compute the two-electron MO integrals.
    ERI_MO = np.einsum('mnlg,gs->mnls', ERI_AO, C)
    ERI_MO = np.einsum('mnls,lr->mnrs', ERI_MO, np.conjugate(C))
    ERI_MO = np.einsum('nq,mnrs->mqrs', C, ERI_MO)
    ERI_MO = np.einsum('mp,mqrs->pqrs', np.conjugate(C), ERI_MO)

    return ERI_MO



def compute_F_SO(wfn, F_MO):
    """
    Compute the spin orbital basis Fock matrix from the MO basis Fock matrix.
    """
    # Compute number of spin orbitals.
    nSO = 2*F_MO.shape[0]

    # Compute the SO Fock matrix.
    F_SO = np.zeros([nSO, nSO])
    F_SO = F_SO.astype('complex128')
    for p in range(0, nSO):
        if p % 2 == 0:
            p_spin = 1
        elif p % 2 != 0:
            p_spin = -1
        for q in range(0, nSO):
            if q % 2 == 0:
                q_spin = 1
            elif q % 2 != 0:
                q_spin = -1

            # Compute the spin integration.
            spin_int = p_spin * q_spin
            if spin_int < 0:
                spin_int = 0

            # Compute spin orbital matrix elements.
            F_SO[p,q] = F_MO[p//2,q//2] * spin_int

    return F_SO



def compute_ERI_SO(wfn, ERI_MO):
    """
    Compute the spin orbital electron repulsion integrals from the MO basis electron repulsion integrals.
    """
    # Compute the number of spin orbitals.
    #nSO = 2 * wfn.nbf
    nSO0 = 2*ERI_MO.shape[0]
    nSO1 = 2*ERI_MO.shape[1]
    nSO2 = 2*ERI_MO.shape[2]
    nSO3 = 2*ERI_MO.shape[3]

    # Compute the SO ERIs.
    ERI_SO = np.zeros([nSO0, nSO1, nSO2, nSO3])
    ERI_SO = ERI_SO.astype('complex128')
    for p in range(0, nSO0):
        if p % 2 == 0:
            p_spin = 1
        elif p % 2 != 0:
            p_spin = -1
        for q in range(0, nSO1):
            if q % 2 == 0:
                q_spin = 1
            elif q % 2 != 0:
                q_spin = -1
            for r in range(0, nSO2):
                if r % 2 == 0:
                    r_spin = 1
                elif r % 2 != 0:
                    r_spin = -1
                for s in range(0, nSO3):
                    if s % 2 == 0:
                        s_spin = 1
                    elif s % 2 != 0:
                        s_spin = -1

                    # Compute the spin integration.
                    spin_int_pq = p_spin * q_spin
                    if spin_int_pq < 0:
                        spin_int_pq = 0
                    spin_int_rs = r_spin * s_spin
                    if spin_int_rs < 0:
                        spin_int_rs = 0
                    spin_int = spin_int_pq * spin_int_rs

                    # Compute spin orbital matrix elements.
                    ERI_SO[p,q,r,s] = ERI_MO[p//2,q//2,r//2,s//2] * spin_int

    return ERI_SO



# Computes the molecular orbital overlap between two wavefunctions.
def compute_mo_overlap(ndocc, nbf, bra_basis, bra_wfn, ket_basis, ket_wfn):
    mints = psi4.core.MintsHelper(bra_basis)

    if bra_basis == ket_basis:
        ao_overlap = mints.ao_overlap().np
    elif bra_basis != ket_basis:
        ao_overlap = mints.ao_overlap(bra_basis, ket_basis).np

    mo_overlap = np.zeros_like(ao_overlap)
    mo_overlap = mo_overlap.astype('complex128')

    for m in range(0, nbf):
        for n in range(0, nbf):
            for mu in range(0, nbf):
                for nu in range(0, nbf):
                    mo_overlap[m, n] += np.conjugate(np.transpose(bra_wfn[mu, m])) *  ao_overlap[mu, nu] * ket_wfn[nu, n]
    return mo_overlap



# Computes the spin orbital overlap from the molecular orbital overlap.
def compute_so_overlap(nbf, mo_overlap):
    """  
    Compute the spin orbital basis overlap matrix from the MO basis overlap matrix.
    """
    # Compute number of spin orbitals.
    nSO = 2 * nbf

    # Compute the SO Fock matrix.
    S_SO = np.zeros([nSO, nSO])
    S_SO = S_SO.astype('complex128')
    for p in range(0, nSO):
        if p % 2 == 0:
            p_spin = 1  
        elif p % 2 != 0:
            p_spin = -1 
        for q in range(0, nSO):
            if q % 2 == 0:
                q_spin = 1  
            elif q % 2 != 0:
                q_spin = -1 

            # Compute the spin integration.
            spin_int = p_spin * q_spin
            if spin_int < 0: 
                spin_int = 0  

            # Compute spin orbital matrix elements.
            S_SO[p,q] = mo_overlap[p//2,q//2] * spin_int

    return S_SO



# Compute MO-level phase correction.
def compute_phase(ndocc, nbf, unperturbed_basis, unperturbed_wfn, ket_basis, ket_wfn):
    # Compute MO overlaps.
    mo_overlap1 = compute_mo_overlap(ndocc, nbf, unperturbed_basis, unperturbed_wfn, ket_basis, ket_wfn)
    mo_overlap2 = np.conjugate(np.transpose(mo_overlap1))

    new_ket_wfn = np.zeros_like(ket_wfn)

    # Compute the phase corrected coefficients.
    for m in range(0, nbf):
        # Compute the normalization.
        N = np.sqrt(mo_overlap1[m][m] * mo_overlap2[m][m])

        # Compute phase factor.
        phase_factor = mo_overlap1[m][m] / N

        # Compute phase corrected overlap.
        for mu in range(0, nbf):
            new_ket_wfn[mu][m] = ket_wfn[mu][m] * (phase_factor ** -1)

    return new_ket_wfn



def line_shape(frequency, intensity, fwhm, number_of_points, min_freq, max_freq):
    """
    Fits the VCD rotatory strengths to a line shape function.
    """
    # Set up physical constants.
    _c = psi4.qcel.constants.get("speed of light in vacuum") # m/s
    _na = psi4.qcel.constants.get("Avogadro constant") # 1/mol
    _h = psi4.qcel.constants.get("Planck constant") # J s

    # Convert to cgs units.
    _c = _c * 10**2 # cm / s
    _h = _h * 10**7 # g cm^2 / s

    # Sorts the frequencies and intensities for a given test in ascending order of frequencies.
    freq, ints = zip(*sorted(zip(frequency, intensity)))

    # Multiplying by rotatory strengths by 10^44 to since it was pulled out previously for reporting when converting to cgs units.
    ints = np.array(ints)
    ints *= (1 / 10**(44))

    # Define the interval at which points will be plotted for the x-coordinate.
    delta = float((max_freq - min_freq)/number_of_points)

    # Obtain the values associated with the x-coordinates in cm-1.
    freq_axis = np.arange(min_freq, max_freq, delta)

    # Initialize the array associated with the y-coordinates.
    ints_axis = np.zeros_like(freq_axis)

    # Compute the intensity associated with the given frequency.
    for a in range(len(freq_axis)):
        for b in range(len(freq)):
            #ints_axis[a] += ints[b]*((0.5 * fwhm)**2/(4*(freq_axis[a]-freq[b])**2+(0.5 * fwhm)**2))
            ints_axis[a] += 10**3 * (32*np.pi**3*_na*freq_axis[a])/(3000*_h*_c*np.log(10)) * ints[b]*(((0.5*fwhm)/np.pi)/((freq[b]-freq_axis[a])**2+(0.5 * fwhm)**2))

    return freq_axis, ints_axis



















