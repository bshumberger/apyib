"""Contains the Hartree-Fock wavefunction object."""

import psi4
import numpy as np
import scipy.linalg as la
import opt_einsum as oe
from apyib.utils import solve_general_DIIS

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
        self.nbf = H.basis_set.nbf()

        # Defining orbital coefficients and orbital energies as properties of the wavefunction. Updated after "solve_SCF".
        self.C = np.zeros((self.nbf,self.nbf))
        self.eps = np.zeros((self.nbf))
        self.E_SCF = 0

    # Computes the number of electrons.
    def nelectron(self, charge):
        nelec = -charge
        for atom in range(self.H.molecule.natom()):
            nelec += self.H.molecule.true_atomic_number(atom)

        return nelec

    # Solve the SCF procedure.
    def solve_SCF(self, parameters, print_level=0):
        """
        Solves the self-consistent field (SCF) procedure with or without DIIS.
        """
        # Compute the initial guess Fock matrix and density matrix.
        # Compute the core Hamiltonian.
        H = self.H
        H_core = H.T + H.V

        # Compute the orthogonalization matrix.
        X = np.linalg.inv(la.sqrtm(H.S))

        # Compute the initial (guess) Fock matrix in the orthonormal AO basis.
        F_p = X @ H_core @ X

        # Diagonalize the initial Fock matrix.
        e, C_p = np.linalg.eigh(F_p)      # F_p C_p = C_p e

        # Transform the eigenvectors into the original AO basis.
        C = X @ C_p

        # Compute the inital density matrix.
        D = 2 * oe.contract('mp,np->mn', C[0:self.nbf,0:self.ndocc], np.conjugate(C[0:self.nbf,0:self.ndocc]))

        # Compute the inital Hartree-Fock Energy
        E_SCF = 0
        for mu in range(self.nbf):
            for nu in range(self.nbf):
                E_SCF += 0.5 * D[mu, nu] * ( H_core[mu, nu] + H_core[mu, nu] )
        E_tot = E_SCF.real + H.E_nuc

        if print_level > 0:
            print("\n Iter      E_elec(real)       E_elec(imaginary)        E(tot)           Delta_E(real)       Delta_E(imaginary)      RMS_D(real)      RMS_D(imaginary)")
            print(" %02d %20.12f %20.12f %20.12f" % (0, E_SCF.real, E_SCF.imag, E_tot))

        # Starting the SCF procedure.
        i = 1
        while i <= parameters['max_iterations']:
            E_old = E_SCF
            D_old = D

            # Solve for the Fock matrix.
            F = H_core + 0.5 * oe.contract('ls,mnls->mn', D, 2 * H.ERI - H.ERI.swapaxes(1,2))

            # Solve DIIS equations.
            if parameters['DIIS']:
                F_flat = len(np.reshape(F, (-1)))
                res_vec = np.reshape(X@(H.S@D@F - np.conjugate(np.transpose(H.S@D@F)))@X, (-1))
                F_vec = np.reshape(F, (-1))
                if i == 1:
                    F_iter = np.atleast_2d(F_vec).T
                    e_iter = np.atleast_2d(res_vec).T
                F_vec, e_iter, F_iter = solve_general_DIIS(parameters, res_vec, F_vec, e_iter, F_iter, i)
                F = np.reshape(F_vec, (self.nbf, self.nbf))

            # Compute the molecular orbital coefficients.
            if i >= 1:
                F_p = X @ F @ X
                self.eps, C_p = np.linalg.eigh(F_p)
                C = self.C = X @ C_p

            # Compute the new density.
            D = 2 * oe.contract('mp,np->mn', C[0:self.nbf,0:self.ndocc], np.conjugate(C[0:self.nbf,0:self.ndocc]))

            # Compute the new energy.
            E_SCF = 0.0
            for mu in range(self.nbf):
                for nu in range(self.nbf):
                    E_SCF += 0.5 * D[nu, mu] * ( H_core[mu, nu] + F[mu, nu] )
            E_tot = E_SCF + H.E_nuc

            # Compute the energy convergence.
            delta_E = E_SCF - E_old
    
            # Compute convergence data.
            rms_D2 = 0
            for mu in range(self.nbf):
                for nu in range(self.nbf):
                    rms_D2 += (D_old[mu, nu] - D[mu, nu])**2
            rms_D = np.sqrt(rms_D2)

            if print_level > 0:
                print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f %20.12f" % (i, E_SCF.real, E_SCF.imag, E_tot, delta_E.real, delta_E.imag, rms_D.real, rms_D.imag))
    
            if i > 1:
                if abs(delta_E) < parameters['e_convergence'] and rms_D < parameters['d_convergence']:
                    #print("Convergence criteria met.")
                    break
            if i == parameters['max_iterations']:
                if abs(delta_E) > parameters['e_convergence'] or rms_D > parameters['d_convergence']:
                    if print_level > 0:
                        print("Not converged.")
    
            i += 1

        self.E_SCF = E_SCF

        #################################################
        ## Compute the AO to MO transformed SCF energy.
        #H_core_MO = oe.contract('ip,ij,jq->pq', np.conjugate(C), H.T + H.V, C)
        #F_MO = oe.contract('ip,ij,jq->pq', np.conjugate(C), F, C)
        #E1 = 0.0 
        #for i in range(0,self.ndocc):
        #    E1 += H_core_MO[i][i] + F_MO[i][i]
        #print('AO to MO Transformed Energy:', E1 + self.H.E_nuc)

        ## Compute AO density based SCF energy.
        #E_SCF1 = oe.contract('vu,uv->', D, H_core + F)
        #print('AO Density-Based Energy:',E_SCF1 + self.H.E_nuc)

        ##print(C,'\n')
        #print('D')
        #print(D,'\n')
        #print('C[0:self.nbf,0:self.ndocc] @ np.conjugate(np.transpose(C)[0:self.ndocc,0:self.nbf])')
        #print(C[0:self.nbf,0:self.ndocc] @ np.conjugate(np.transpose(C)[0:self.ndocc,0:self.nbf]), '\n')
        #print('C[0:self.nbf,0:self.ndocc]')
        #print(C[0:self.nbf,0:self.ndocc], '\n')
        #print('np.conjugate(np.transpose(C)[0:self.ndocc,0:self.nbf])')
        #print(np.conjugate(np.transpose(C)[0:self.ndocc,0:self.nbf]), '\n')

        # Test MO Density
        #print(np.conjugate(np.transpose(C))@H.S@D@H.S@C)

        # Test Idempotency
        #print(D - (D@H.S@D))
        #################################################


        return E_SCF, self.C


