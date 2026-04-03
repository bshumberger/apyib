"""This script contains a set of functions for analytic evaluation of the Hessian."""

import numpy as np
import psi4
import gc
import opt_einsum as oe
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn
from apyib.utils import get_slices
from apyib.utils import compute_ERI_MO
from apyib.utils import solve_general_DIIS

class analytic_derivative(object):
    """ 
    Analytic derivative object.
    """
    # Defines the integrals associated with the analytic evaluation of the energy.
    def __init__(self, parameters):
        # Set calculation parameters.
        self.parameters = parameters

        # Perform RHF energy calculation.
        self.H = Hamiltonian(self.parameters)
        self.wfn = hf_wfn(self.H)
        E_SCF, self.C = self.wfn.solve_SCF(self.parameters)



    def compute_RHF_momentum_Hessian(self, orbitals='non-canonical'):
        # Setting initial variables for readability.
        C = self.C
        nbf = self.wfn.nbf
        no = self.wfn.ndocc
        nv = self.wfn.nbf - self.wfn.ndocc

        # Setting up slices.
        C_list, I_list = get_slices(self.parameters, self.wfn)
        f_ = C_list[0]
        o_ = C_list[1]
        v_ = C_list[2]
        t_ = C_list[3]

        o = slice(0, no) 
        v = slice(no, nbf)
        t = slice(0, nbf)

        # Create a Psi4 matrix object for obtaining the perturbed MO basis integrals.
        C_p4 = psi4.core.Matrix.from_array(C)

        # Set the atom lists for Hessian.
        natom = self.H.molecule.natom()
        atoms = np.arange(0, natom)

        # Compute the core Hamiltonian in the MO basis.
        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)

        # Compute the electron repulsion integrals in the MO basis.
        ERI = compute_ERI_MO(self.parameters, self.wfn, C_list)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Computing the Fock matrix excluding any terms dependent on the second derivative coupling.
        mints = psi4.core.MintsHelper(self.H.basis_set)
        V_bare = mints.ao_potential().np
        F_bare = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + V_bare,C) + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Compute Gamma and Gamma tilde in the MO basis.
        Gamma = oe.contract('mp,abmn,nq->abpq', np.conjugate(C), self.H.G, C)
        Zeta_tilde = oe.contract('mp,abmn,nq->abpq', np.conjugate(C), self.H.zeta_tilde, C)

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)

        # Set up the momentum Hessian.
        momentum_Hessian = np.zeros((natom * 3, natom * 3))

        # Set up a list of CPHF matrices for each perturbation to keep track of perturbations and avoid duplicate computations.
        U_P = []
        U_P_list = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        for N1 in atoms:
            for a in range(3):
                N1a = 3 * N1 + a
                if N1a in U_P_list:
                    ind = U_P_list.index(N1a)
                    U_Pa = U_P[ind]

                else:
                    # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                    B = Gamma[N1][a][v,o] / self.H.M[N1]

                    # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                    U_Pa = np.zeros((nbf,nbf))
                    U_Pa[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                    U_Pa[o,v] += U_Pa[v,o].T

                    # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                    if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                        D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                        B = - (Gamma[N1][a][o,o] / self.H.M[N1]) + oe.contract('em,iejm->ij', U_Pa[v,o], A.swapaxes(1,2)[o,v,o,o])
                        U_Pa[o,o] += B/D

                        D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                        B = - (Gamma[N1][a][v,v] / self.H.M[N1]) + oe.contract('em,aebm->ab', U_Pa[v,o], A.swapaxes(1,2)[v,v,v,o])
                        U_Pa[v,v] += B/D

                        for j in range(no):
                            U_Pa[j,j] = 0
                        for c in range(no,nbf):
                            U_Pa[c,c] = 0

                    if orbitals == 'non-canonical':
                        U_Pa[f_,f_] = 0
                        U_Pa[o_,o_] = 0
                        U_Pa[v_,v_] = 0

                    # Appending CPHF matrices to list.
                    U_P.append(U_Pa)
                    U_P_list.append(N1a)

                for N2 in range(N1, natom):
                    for b in range(3):
                        ab = 3 * a + b 
                        N2b = 3 * N2 + b
                        
                        if N2b < N1a:
                            momentum_Hessian[N1a][N2b] = 0

                        elif N2b >= N1a:
                            if N2b in U_P_list:
                                ind = U_P_list.index(N2b)
                                U_Pb = U_P[ind]

                            else:
                                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                                B = Gamma[N2][b][v,o] / self.H.M[N2]

                                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                                U_Pb = np.zeros((nbf,nbf))
                                U_Pb[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                                U_Pb[o,v] += U_Pb[v,o].T

                                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                                    B = - (Gamma[N2][b][o,o] / self.H.M[N2]) + oe.contract('em,iejm->ij', U_Pb[v,o], A.swapaxes(1,2)[o,v,o,o])
                                    U_Pb[o,o] += B/D

                                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                                    B = - (Gamma[N2][b][v,v] / self.H.M[N2]) + oe.contract('em,aebm->ab', U_Pb[v,o], A.swapaxes(1,2)[v,v,v,o]) 
                                    U_Pb[v,v] += B/D

                                    for j in range(no):
                                        U_Pb[j,j] = 0
                                    for c in range(no,nbf):
                                        U_Pb[c,c] = 0

                                if orbitals == 'non-canonical':
                                    U_Pb[f_,f_] = 0
                                    U_Pb[o_,o_] = 0
                                    U_Pb[v_,v_] = 0

                                # Appending CPHF matrices to list.
                                U_P.append(U_Pb)
                                U_P_list.append(N2b)

                            # Symmetric approach to momentum Hessian calculation. Note that all the signs on the equations are changed because these are squared
                            # imaginary quantities.
                            if N1a == N2b:
                                momentum_Hessian[N1a][N2b] += 1 / self.H.M[N1]
                            momentum_Hessian[N1a][N2b] -= (4 / self.H.M[N1]) * oe.contract('pi,pi->', U_Pb[:,o], Gamma[N1][a][:,o])
                            momentum_Hessian[N1a][N2b] -= (4 / self.H.M[N2]) * oe.contract('pi,pi->', U_Pa[:,o], Gamma[N2][b][:,o])
                            momentum_Hessian[N1a][N2b] += 4 * oe.contract('pi,qi,c,cgpq->', U_Pa[:,o], U_Pb[:,o], np.reciprocal(2 * self.H.M), Zeta_tilde)
                            momentum_Hessian[N1a][N2b] += 4 * oe.contract('pi,qi, pq->', U_Pa[:,o], U_Pb[:,o], F_bare[:,:])
                            momentum_Hessian[N1a][N2b] -= 4 * oe.contract('pi,pi, i->', U_Pa[:,o], U_Pb[:,o], self.wfn.eps[o])
                            momentum_Hessian[N1a][N2b] -= 4 * oe.contract('pi,qj,pqij->', U_Pa[:,o], U_Pb[:,o], 2 * ERI[:,:,o,o] - 2 * ERI[:,o,o,:].swapaxes(1,3) - ERI[:,:,o,o].swapaxes(2,3) + ERI[:,o,:,o].swapaxes(1,2).swapaxes(2,3))
                            #momentum_Hessian[N1a][N2b] -= 4 * oe.contract('pi,qj,pqij->', U_Pa[:,o], U_Pb[:,o], - ERI[:,:,o,o].swapaxes(2,3) + ERI[:,o,:,o].swapaxes(1,2).swapaxes(2,3))
                            # Antisymmetric approach to momentum Hessian calculation.
                            #if N1a == N2b:
                            #    momentum_Hessian[N1a][N2b] += 1 / self.H.M[N1]
                            #momentum_Hessian[N1a][N2b] -= (4 / self.H.M[N1]) * oe.contract('pi,pi->', U_Pb[:,o], Gamma[N1][a][:,o])

        momentum_Hessian += momentum_Hessian.T
        momentum_Hessian -= 0.5 * np.eye(3*natom) * momentum_Hessian

        return momentum_Hessian













