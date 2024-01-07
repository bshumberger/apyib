"""Contains the Hartree-Fock wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
from apyib.utils import solve_DIIS

class hf_wfn(object):
    """
    Wavefunction object.
    """
    # Define the specific properties of the Hartree-Fock wavefunction.
    def __init__(self, H, charge=0):

        # Define the Hamiltonian and number of doubly occupied orbitals as properties of the wavefunction.
        self.H = H
        self.nelec = self.nelectron(charge)
        self.ndocc = self.nelec // 2
        self.nbf = H.S.shape[0]

    # Computes the number of electrons.
    def nelectron(self, charge):
        nelec = -charge
        for atom in range(self.H.molecule.natom()):
            nelec += self.H.molecule.true_atomic_number(atom)

        return nelec

    # Solve the SCF procedure.
    def solve_SCF(self, parameters):
        """
        Solves the self-consistent field (SCF) procedure with or without DIIS.
        """
        # Compute the initial guess Fock matrix and density matrix.
        # Compute the core Hamiltonian.
        H = self.H
        H_core = H.T + H.V
        H_core = H_core.astype('complex128')

        # Add electric and magnetic potentials to the core Hamiltonian.
        for alpha in range(3):
            H_core -=  parameters['F_el'][alpha] * H.mu_el[alpha] + parameters['F_mag'][alpha] * H.mu_mag[alpha]

        # Compute the orthogonalization matrix.
        X = np.linalg.inv(la.sqrtm(H.S))

        # Compute the initial (guess) Fock matrix in the orthonormal AO basis.
        F_p = X @ H_core @ X

        # Diagonalize the initial Fock matrix.
        e, C_p = np.linalg.eigh(F_p)      # F_p C_p = C_p e

        # Transform the eigenvectors into the original AO basis.
        C = X @ C_p

        # Compute the inital density matrix.
        #D = np.zeros_like(C)
        #for mu in range(self.nbf):
        #    for nu in range(self.nbf):
        #        for m in range(self.ndocc):
        #            D[mu, nu] += C[mu, m] * np.conjugate(np.transpose(C[nu, m]))

        D = np.einsum('mp,np->mn', C[0:self.nbf,0:self.ndocc], np.conjugate(C[0:self.nbf,0:self.ndocc]))

        # Compute the inital Hartree-Fock Energy
        E_SCF = 0
        for mu in range(self.nbf):
            for nu in range(self.nbf):
                E_SCF += D[mu, nu] * ( H_core[mu, nu] + H_core[mu, nu] )
        E_tot = E_SCF.real + H.E_nuc

        #print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_D(real)      RMS_D(imaginary)")
        #print(" %02d %20.12f %20.12f %20.12f" % (0, E_SCF.real, E_SCF.imag, E_tot))

        # Starting the SCF procedure.
        # Setting up DIIS arrays for the error matrices and Fock matrices.
        if parameters['DIIS']:
            e_iter = []
            F_iter = []

        i = 1
        while i <= parameters['max_iterations']:
            E_old = E_SCF
            D_old = D
            #F = np.zeros_like(F_p)
            #for mu in range(self.nbf):
            #    for nu in range(self.nbf):
            #        F[mu, nu] += H_core[mu, nu]
            #        for lambd in range(self.nbf):
            #            for sigma in range(self.nbf):
            #                F[mu, nu] += D[lambd, sigma] * ( 2 * H.ERI[mu, nu, lambd, sigma] - H.ERI[mu, lambd, nu, sigma] )

            F = H_core + np.einsum('ls,mnls->mn', D, 2 * H.ERI - H.ERI.swapaxes(1,2))

            # Solve DIIS equations.
            if parameters['DIIS']:
                F = solve_DIIS(parameters, F, D, H.S, X, F_iter, e_iter) 

            if i >= 1:
                # Compute molecular orbital coefficients.
                F_p = X @ F @ X
                e, C_p = np.linalg.eigh(F_p)
                C = X @ C_p

            # Compute the new density.
            #D = np.zeros_like(D)
            #for mu in range(self.nbf):
            #    for nu in range(self.nbf):
            #        for m in range(self.ndocc):
            #            D[mu, nu] += C[mu, m] * np.conjugate(np.transpose(C[nu, m]))

            D = np.einsum('mp,np->mn', C[0:self.nbf,0:self.ndocc], np.conjugate(C[0:self.nbf,0:self.ndocc]))

            # Compute the new energy.
            E_SCF = 0.0
            for mu in range(self.nbf):
                for nu in range(self.nbf):
                    E_SCF += D[mu, nu] * ( H_core[mu, nu] + F[mu, nu] )
            E_tot = E_SCF.real + H.E_nuc
    
            # Compute the energy convergence.
            delta_E = E_SCF - E_old
    
            # Compute convergence data.
            rms_D2 = 0
            for mu in range(self.nbf):
                for nu in range(self.nbf):
                    rms_D2 += (D_old[mu, nu] - D[mu, nu])**2
            rms_D = np.sqrt(rms_D2)
            #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (i, E_SCF.real, E_SCF.imag, E_tot, delta_E.real, delta_E.imag, rms_D.real, rms_D.imag))
    
            if i > 1:
                if abs(delta_E) < parameters['e_convergence'] and rms_D < parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if i == parameters['max_iterations']:
                if abs(delta_E) > parameters['e_convergence'] or rms_D > parameters['d_convergence']:
                    print("Not converged.")
    
            i += 1

        return e, E_SCF, E_tot, C


