"""This script contains a set of functions for analytic evaluation of the Hessian."""

import numpy as np
import psi4
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn

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



    def compute_RHF_Hessian(self):
        # Setting initial variables for readability.
        C = self.C
        eps = self.wfn.eps
        nbf = self.wfn.nbf
        no = self.wfn.ndocc
        nv = self.wfn.nbf - self.wfn.ndocc

        # Setting up slices.
        o = slice(0, no)
        v = slice(no, nbf)
        t = slice(0, nbf)

        # Create a Psi4 matrix object for obtaining the perturbed MO basis integrals.
        C_p4 = psi4.core.Matrix.from_array(C)
        
        # Set the atom lists for Hessian.
        natom = self.H.molecule.natom()
        atoms = np.arange(0, natom)

        # Compute the core Hamiltonian in the MO basis.
        h = np.einsum('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)

        # Compute the electron repulsion integrals in the MO basis.
        ERI = np.einsum('mnlg,gs->mnls', self.H.ERI, C)
        ERI = np.einsum('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = np.einsum('nq,mnrs->mqrs', C, ERI)
        ERI = np.einsum('mp,mqrs->pqrs', np.conjugate(C), ERI)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + np.einsum('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Compute the perturbation-independent A matrix for the CPHF coefficients.
        A = 4 * ERI - ERI.swapaxes(1,2) - ERI.swapaxes(2,3)
        A = A.swapaxes(1,2)
        G = np.einsum('ik,jl,ijkl->ijkl', np.eye(nbf), np.eye(nbf), F.reshape(1,nbf,1,nbf) - F.reshape(nbf,1,nbf,1)) + A
        #G_inv = np.linalg.tensorinv(G[o,v,o,v])

        # First derivative matrices.
        h_a = []
        S_a = []
        F_a = []
        ERI_a = []
        U_a = []

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_d1 = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_d1 = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_d1 = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_d1 = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_d1[a] = T_d1[a].np
                V_d1[a] = V_d1[a].np
                S_d1[a] = S_d1[a].np
                ERI_d1[a] = ERI_d1[a].np
                ERI_d1[a] = ERI_d1[a].swapaxes(1,2)

                # Computing skeleton (core) first derivative integrals.
                h_d1 = T_d1[a] + V_d1[a]
                F_d1 = T_d1[a] + V_d1[a] + np.einsum('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B_d1 = -F_d1 + np.einsum('ij,j->ij', S_d1[a], eps) + np.einsum('kl,ikjl->ij', S_d1[a][o,o], 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

                # Solve for the CPHF coefficients.
                U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
                U_d1[o,o] -= 0.5 * S_d1[a][o,o]
                #U_d1[v,o] += np.einsum('ijkl,lk->ji', G_inv, B_d1[v,o])
                U_d1[v,o] += np.linalg.tensorsolve(G[o,v,o,v], B_d1.T[o,v]).T
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]
                U_d1[v,v] -= 0.5 * S_d1[a][v,v]
                #print("U Matrix:")
                #print(U_d1,"\n")

                # Appending to lists.
                S_a.append(S_d1[a])
                ERI_a.append(ERI_d1[a])
                h_a.append(h_d1)
                F_a.append(F_d1)
                U_a.append(U_d1)

        # Second derivative matrices.
        h_ab = []
        S_ab = []
        F_ab = []
        ERI_ab = []
        
        # Compute and store second derivative integrals.
        for N1 in atoms:
            for N2 in atoms:
                # Compute the skeleton (core) one-electron second derivative integrals in the MO basis.
                T_d2 = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_d2 = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_d2 = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)

                # Compute the skeleton (core) two-electron second derivative integrals in the MO basis.
                ERI_d2 = mints.mo_tei_deriv2(N1, N2, C_p4, C_p4, C_p4, C_p4)

                for ab in range(9):
                    # Convert the Psi4 matrices to numpy matrices.
                    T_d2[ab] = T_d2[ab].np
                    V_d2[ab] = V_d2[ab].np
                    S_d2[ab] = S_d2[ab].np
                    ERI_d2[ab] = ERI_d2[ab].np
                    ERI_d2[ab] = ERI_d2[ab].swapaxes(1,2)

                    # Computing and appending the skeleton (core) second derivative integrals.
                    S_ab.append(S_d2[ab])
                    ERI_ab.append(ERI_d2[ab])
                    h_ab.append(T_d2[ab] + V_d2[ab])
                    F_ab.append(T_d2[ab] + V_d2[ab] + np.einsum('piqi->pq', 2 * ERI_d2[ab][:,o,:,o] - ERI_d2[ab].swapaxes(2,3)[:,o,:,o]))

        for N1 in atoms:
            for N2 in atoms:
                for a in range(3):
                    for b in range(3):
                        # Defining common index for the first and second derivative components with respect to the Hessian index.
                        alpha = 3*N1 + a
                        beta = 3*N2 + b
                        N_ab = N1*natom*9 + N2*9 + a*3 + b%3

                        # Computing the eta matrix.
                        eta_ab = np.einsum('im,jm->ij', U_a[alpha][o,:], U_a[beta][o,:]) + np.einsum('im,jm->ij', U_a[beta][o,:], U_a[alpha][o,:]) - np.einsum('im,jm->ij', S_a[alpha][o,:], S_a[beta][o,:]) - np.einsum('im,jm->ij', S_a[beta][o,:], S_a[alpha][o,:]) 

                        # Computing the Hessian.
                        Hessian[alpha][beta] += 2 * np.einsum('ii->', h_ab[N_ab][o,o]) + np.einsum('ijij->', 2 * ERI_ab[N_ab][o,o,o,o] - ERI_ab[N_ab].swapaxes(2,3)[o,o,o,o])
                        Hessian[alpha][beta] -= 2 * np.einsum('ii,i->', S_ab[N_ab][o,o], eps[o])
                        Hessian[alpha][beta] -= 2 * np.einsum('ii,i->', eta_ab, eps[o])
                        Hessian[alpha][beta] += 4 * np.einsum('mj,mj->', U_a[beta][:,o], F_a[alpha][:,o]) + 4 * np.einsum('mj,mj->', U_a[alpha][:,o], F_a[beta][:,o])
                        Hessian[alpha][beta] += 4 * np.einsum('mj,mj,m->', U_a[alpha][:,o], U_a[beta][:,o], eps)
                        Hessian[alpha][beta] += 4 * np.einsum('mj,nl,mjnl->', U_a[alpha][:,o], U_a[beta][:,o], A[:,o,:,o])

        Nuc_Hessian = self.H.molecule.nuclear_repulsion_energy_deriv2().np
        Hessian += Nuc_Hessian

        print(Hessian)

        return Hessian












