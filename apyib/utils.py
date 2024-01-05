"""This script contains a set of functions for comparisons and improved performance."""

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





