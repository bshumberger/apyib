"""This script contains a set of functions for comparisons, improved performance, and regular operations associated with multiple methods."""

import numpy as np
import psi4



def run_psi4(parameters, method='RHF'):
    """ 
    Run Psi4 for comparison to the apyib code. Note that psi4 does not do magnetic field perturbations.
    """
    # Set the name of the output file.
    psi4.set_output_file("output.dat", False)

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
    B = B.astype('complex128')
    for m in range(len(e_iter)):
        for n in range(len(e_iter)):
            #B[m,n] = np.einsum('mn, mn->',e_iter[m], e_iter[n], optimize=True)
            for o in range(len(e_diis[0])):
                for p in range(len(e_diis[0])):
                    B[m,n] += e_iter[m][o][p] * e_iter[n][o][p]
    
    # Build the "A" matrix for the system of linear equations.
    A = np.zeros(len(e_iter)+1)
    A[-1] = -1
    A = A.astype('complex128')
    
    # Solve the system of linear equations.
    #C_diis = np.linalg.inv(B) @ A
    C_diis = np.linalg.solve(B,A)
    
    # Solve for new Fock matrix.
    F = np.zeros_like(F)
    for j in range(len(C_diis) - 1): 
        F += C_diis[j] * F_iter[j]

    return F



def compute_F_MO(parameters, H, wfn, C):
    """
    Computes the MO basis Fock matrix from the AO basis.
    """
    # Set up the one-electron AO integrals.
    h_AO = H.T + H.V

    # Set up the two-electron AO integrals.
    ERI_AO = H.ERI.astype('complex128')

    # Compute the density.
    D = np.einsum('mp,np->mn', C[0:wfn.nbf,0:wfn.ndocc], np.conjugate(C[0:wfn.nbf,0:wfn.ndocc]))

    # Compute the Fock matrix elements.
    F_AO = h_AO + np.einsum('ls,mnls->mn', D, 2 * ERI_AO - ERI_AO.swapaxes(1,2))

    # Compute MO Fock matrix elements.
    F_MO = np.einsum('ip,ij,jq->pq', np.conjugate(C), F_AO, C)

    return F_MO



def compute_ERI_MO(parameters, H, wfn, C):
    """
    Computes the MO basis electron repulsion integrals from the AO basis.
    """
    # Set up the two-electron AO integrals.
    ERI_AO = H.ERI.astype('complex128')

    # Compute the two-electron MO integrals.
    ERI_MO = np.einsum('mnlg,gs->mnls', H.ERI, C)
    ERI_MO = np.einsum('mnls,lr->mnrs', ERI_MO, np.conjugate(C))
    ERI_MO = np.einsum('nq,mnrs->mqrs', C, ERI_MO)
    ERI_MO = np.einsum('mp,mqrs->pqrs', np.conjugate(C), ERI_MO)

    return ERI_MO



def compute_F_SO(wfn, F_MO):
    """
    Compute the spin orbital basis Fock matrix from the MO basis Fock matrix.
    """
    # Compute number of spin orbitals.
    nSO = 2 * wfn.nbf

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



def line_shape(frequency, intensity, fwhm, number_of_points, min_freq, max_freq):
    """
    Fits the VCD rotatory strengths to a line shape function.
    """
    # Sorts the frequencies and intensities for a given test in ascending order of frequencies.
    freq, ints = zip(*sorted(zip(frequency, intensity)))

    # Define the interval at which points will be plotted for the x-coordinate.
    delta = float((max_freq - min_freq)/number_of_points)

    # Obtain the values associated with the x-coordinates in cm-1.
    freq_axis = np.arange(min_freq, max_freq, delta)

    # Initialize the array associated with the y-coordinates.
    ints_axis = np.zeros_like(freq_axis)

    # Compute the intensity associated with the given frequency.
    for a in range(len(freq_axis)):
        for b in range(len(freq)):
            ints_axis[a] += ints[b]*((0.5 * fwhm)**2/(4*(freq_axis[a]-freq[b])**2+(0.5 * fwhm)**2))

    return freq_axis, ints_axis



















