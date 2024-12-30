"""This script contains a set of functions for analytic evaluation of the Hessian."""

import numpy as np
import psi4
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn

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

        # Set up the gradient.
        Gradient = np.zeros((natom, 3))

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

                # Computing the gradient.
                Gradient[N1][a] = 2 * np.einsum('ii->', h_d1[o,o])
                Gradient[N1][a] += np.einsum('ijij->', 2 * ERI_d1[a][o,o,o,o] - ERI_d1[a].swapaxes(2,3)[o,o,o,o])
                Gradient[N1][a] -= 2 * np.einsum('ii,i->', S_d1[a][o,o], eps[o])

        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np
        Gradient += Nuc_Gradient
        #print("Gradient:")
        #print(Gradient)

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
        #print("Hessian:")
        #print(Hessian)

        return Gradient, Hessian



    def compute_MP2_Hessian(self):
        # Compute RHF Hessian for addition to MP2 contribution.
        rhf_gradient, rhf_hessian = self.compute_RHF_Hessian()

        # Compute T2 amplitudes and MP2 energy.
        wfn_MP2 = mp2_wfn(self.parameters, self.wfn)
        E_MP2, t2 = wfn_MP2.solve_MP2()

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

        # Get nuclear repulsion energy gradient.
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)

        # Set up the gradient.
        Gradient = np.zeros((natom, 3))

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

                ### Computing the gradient. If the geometry is optimized, the gradient should be zero.
                f_grad = F_d1.astype('complex128')
                f_grad += np.einsum('rp,rq->pq', U_d1, F) + np.einsum('rq,pr->pq', U_d1, F)
                ###f_grad += np.einsum('ri,prqi->pq', U_d1[:,o], 2*ERI[:,:,:,o] - ERI.swapaxes(2,3)[:,:,:,o])
                ###f_grad += np.einsum('ri,piqr->pq', U_d1[:,o], 2*ERI[:,o,:,:] - ERI.swapaxes(2,3)[:,o,:,:])
                f_grad += np.einsum('ri,prqi->pq', U_d1[:,o], A.swapaxes(1,2)[:,:,:,o])
                #print("Fock Matrix Derivative", f_grad, "\n")
                ERI_grad = ERI_d1[a][o,o,v,v].astype('complex128')
                ERI_grad += np.einsum('pi,pjab->ijab', U_d1[:,o], ERI[:,o,v,v])
                ERI_grad += np.einsum('pj,ipab->ijab', U_d1[:,o], ERI[o,:,v,v])
                ERI_grad += np.einsum('pa,ijpb->ijab', U_d1[:,v], ERI[o,o,:,v])
                ERI_grad += np.einsum('pb,ijap->ijab', U_d1[:,v], ERI[o,o,v,:])
                #print("ERI_a", ERI_grad, "\n") # This term matches Kirk's code.
                E_grad = np.einsum('ijab,ijab->', 2*(2*t2-t2.swapaxes(2,3)), ERI_grad).astype('complex128')
                E_grad -= np.einsum('ijab,kjab,ik->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[o,o])
                E_grad -= np.einsum('ijab,ikab,jk->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[o,o])
                E_grad += np.einsum('ijab,ijcb,ac->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[v,v])
                E_grad += np.einsum('ijab,ijac,bc->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[v,v])
                #print("Energy Gradient:", E_grad + rhf_gradient[N1][a]) # This is giving the gradient minus the HF component. Correct!
                t2_grad = ERI_grad.copy()
                t2_grad -= np.einsum('kjab,ik->ijab', t2, f_grad[o,o])
                t2_grad -= np.einsum('ikab,kj->ijab', t2, f_grad[o,o])
                t2_grad += np.einsum('ijcb,ac->ijab', t2, f_grad[v,v])
                t2_grad += np.einsum('ijac,cb->ijab', t2, f_grad[v,v])
                t2_grad /= (wfn_MP2.D_ijab)
                #print("t2 grad", t2_grad, "\n") # This term matches Kirk's code.

                E_grad1 = np.einsum('ijab,ijab->', t2_grad, (2*ERI[o,o,v,v]-ERI.swapaxes(2,3)[o,o,v,v])).astype('complex128')
                E_grad1 += np.einsum('ijab,ijab->', t2, (2*ERI_grad-ERI_grad.swapaxes(2,3)))
                #print("Energy Gradient:", E_grad1 + rhf_gradient[N1][a], "\n") # This matches the other expression for the gradient.

                Gradient[N1][a] = E_grad1

        # Second derivative matrices.
        h_ab = []
        S_ab = []
        F_ab = []
        ERI_ab = []

        ## Compute and store second derivative integrals.
        #for N1 in atoms:
        #    for N2 in atoms:
        #        # Compute the skeleton (core) one-electron second derivative integrals in the MO basis.
        #        T_d2 = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
        #        V_d2 = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
        #        S_d2 = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)

        #        # Compute the skeleton (core) two-electron second derivative integrals in the MO basis.
        #        ERI_d2 = mints.mo_tei_deriv2(N1, N2, C_p4, C_p4, C_p4, C_p4)

        #        for ab in range(9):
        #            # Convert the Psi4 matrices to numpy matrices.
        #            T_d2[ab] = T_d2[ab].np
        #            V_d2[ab] = V_d2[ab].np
        #            S_d2[ab] = S_d2[ab].np
        #            ERI_d2[ab] = ERI_d2[ab].np
        #            ERI_d2[ab] = ERI_d2[ab].swapaxes(1,2)

        #            # Computing and appending the skeleton (core) second derivative integrals.
        #            S_ab.append(S_d2[ab])
        #            ERI_ab.append(ERI_d2[ab])
        #            h_ab.append(T_d2[ab] + V_d2[ab])
        #            F_ab.append(T_d2[ab] + V_d2[ab] + np.einsum('piqi->pq', 2 * ERI_d2[ab][:,o,:,o] - ERI_d2[ab].swapaxes(2,3)[:,o,:,o]))

        #for N1 in atoms:
        #    for N2 in atoms:
        #        for a in range(3):
        #            for b in range(3):
        #                # Defining common index for the first and second derivative components with respect to the Hessian index.
        #                alpha = 3*N1 + a
        #                beta = 3*N2 + b
        #                N_ab = N1*natom*9 + N2*9 + a*3 + b%3

        #                # Computing the eta matrix.
        #                eta_ab = np.einsum('im,jm->ij', U_a[alpha][o,:], U_a[beta][o,:]) + np.einsum('im,jm->ij', U_a[beta][o,:], U_a[alpha][o,:]) - np.einsum('im,jm->ij', S_a[alpha][o,:], S_a[beta][o,:]) - np.einsum('im,jm->ij', S_a[beta][o,:], S_a[alpha][o,:])

        #                # Computing the Hessian.
        #                Hessian[alpha][beta] += 2 * np.einsum('ii->', h_ab[N_ab][o,o]) + np.einsum('ijij->', 2 * ERI_ab[N_ab][o,o,o,o] - ERI_ab[N_ab].swapaxes(2,3)[o,o,o,o])
        #                Hessian[alpha][beta] -= 2 * np.einsum('ii,i->', S_ab[N_ab][o,o], eps[o])
        #                Hessian[alpha][beta] -= 2 * np.einsum('ii,i->', eta_ab, eps[o])
        #                Hessian[alpha][beta] += 4 * np.einsum('mj,mj->', U_a[beta][:,o], F_a[alpha][:,o]) + 4 * np.einsum('mj,mj->', U_a[alpha][:,o], F_a[beta][:,o])
        #                Hessian[alpha][beta] += 4 * np.einsum('mj,mj,m->', U_a[alpha][:,o], U_a[beta][:,o], eps)
        #                Hessian[alpha][beta] += 4 * np.einsum('mj,nl,mjnl->', U_a[alpha][:,o], U_a[beta][:,o], A[:,o,:,o])

        #Nuc_Hessian = self.H.molecule.nuclear_repulsion_energy_deriv2().np
        #Hessian += Nuc_Hessian

        Gradient += rhf_gradient
        print("Gradient:")
        print(Gradient)
        #print(Hessian)

        return Gradient, Hessian



    def compute_RHF_Hessian_Canonical(self):
        # Setting initial variables for readability.
        C = self.C
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

        # Set up the gradient.
        Gradient = np.zeros((natom, 3)) 

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3)) 

        # Compute the perturbation-independent A matrix for the CPHF coefficients.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))
        
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
                B = -F_d1[v,o] + np.einsum('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * np.einsum('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = F_d1[o,o].copy().astype('complex128') - np.einsum('ij,jj->ij', S_d1[a][o,o], F[o,o]) + np.einsum('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = F_d1[v,v].copy().astype('complex128') - np.einsum('ab,bb->ab', S_d1[a][v,v], F[v,v]) + np.einsum('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * np.einsum('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                U_d1[v,v] += B/D

                for j in range(no):
                    U_d1[j,j] = -0.5 * S_d1[a][j,j]
                for b in range(nv,nbf):
                    U_d1[b,b] = -0.5 * S_d1[a][b,b]

                # Appending to lists.
                S_a.append(S_d1[a])
                ERI_a.append(ERI_d1[a])
                h_a.append(h_d1)
                F_a.append(F_d1)
                U_a.append(U_d1)

                occ_eps = self.wfn.eps[o].reshape(-1,1) - self.wfn.eps[o]
                vir_eps = self.wfn.eps[v].reshape(-1,1) - self.wfn.eps[v]

                f_grad = np.zeros((nbf,nbf), dtype='complex128')

                f_grad[o,o] += F_d1[o,o].copy().astype('complex128')
                f_grad[o,o] += U_d1[o,o] * occ_eps
                f_grad[o,o] -= np.einsum('ij,jj->ij', S_d1[a][o,o], F[o,o])
                f_grad[o,o] += np.einsum('em,iejm->ij', U_d1[v,o], (2*ERI[o,v,o,o] - ERI.swapaxes(2,3)[o,v,o,o]) + (2*ERI[o,o,o,v] - ERI.swapaxes(2,3)[o,o,o,v]).swapaxes(1,3))
                f_grad[o,o] -= 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], (2*ERI[o,o,o,o] - ERI.swapaxes(2,3)[o,o,o,o]) + (2*ERI[o,o,o,o] - ERI.swapaxes(2,3)[o,o,o,o]).swapaxes(1,3))
                
                for i in range(no):
                    f_grad[i,i] = F_d1[i,i]
                    f_grad[i,i] -= S_d1[a][i,i] * F[i,i]
                    f_grad[i,i] += np.einsum('em,em->', U_d1[v,o], 2*ERI[i,v,i,o] - ERI.swapaxes(2,3)[i,v,i,o])
                    f_grad[i,i] += np.einsum('em,me->', U_d1[v,o], 2*ERI[i,o,i,v] - ERI.swapaxes(2,3)[i,o,i,v])
                    f_grad[i,i] -= 0.5 * np.einsum('mn,mn->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])
                    f_grad[i,i] -= 0.5 * np.einsum('mn,nm->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])

                f_grad[v,v] += F_d1[v,v].copy().astype('complex128')
                f_grad[v,v] += U_d1[v,v] * vir_eps
                f_grad[v,v] -= np.einsum('ab,bb->ab', S_d1[a][v,v], F[v,v])
                f_grad[v,v] += np.einsum('em,aebm->ab', U_d1[v,o], (2*ERI[v,v,v,o] - ERI.swapaxes(2,3)[v,v,v,o]) + (2*ERI[v,o,v,v] - ERI.swapaxes(2,3)[v,o,v,v]).swapaxes(1,3))
                f_grad[v,v] -= 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], (2*ERI[v,o,v,o] - ERI.swapaxes(2,3)[v,o,v,o]) + (2*ERI[v,o,v,o] - ERI.swapaxes(2,3)[v,o,v,o]).swapaxes(1,3))
                    
                for b in range(nv):
                    b += no
                    f_grad[b,b] = F_d1[b,b]
                    f_grad[b,b] -= S_d1[a][b,b] * F[b,b]
                    f_grad[b,b] += np.einsum('em,em->', U_d1[v,o], 2*ERI[b,v,b,o] - ERI.swapaxes(2,3)[b,v,b,o])
                    f_grad[b,b] += np.einsum('em,me->', U_d1[v,o], 2*ERI[b,o,b,v] - ERI.swapaxes(2,3)[b,o,b,v])
                    f_grad[b,b] -= 0.5 * np.einsum('mn,mn->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])
                    f_grad[b,b] -= 0.5 * np.einsum('mn,nm->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])

                #print("Fock Matrix Derivative:")
                #print(f_grad, "\n")

                # Computing the gradient.
                Gradient[N1][a] = 2 * np.einsum('ii->', h_d1[o,o])
                Gradient[N1][a] += np.einsum('ijij->', 2 * ERI_d1[a][o,o,o,o] - ERI_d1[a].swapaxes(2,3)[o,o,o,o])
                Gradient[N1][a] -= 2 * np.einsum('ii,ii->', S_d1[a][o,o], F[o,o])

        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np
        Gradient += Nuc_Gradient
        print("Gradient:")
        print(Gradient)

        # Second derivative matrices.
        h_ab = []
        S_ab = []
        F_ab = []
        ERI_ab = []

        # Redefining epsilon.
        eps = self.wfn.eps

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
        print("Hessian:")
        print(Hessian)

        return Gradient, Hessian



    def compute_MP2_AATs_Canonical(self, normalization='full'):
        # Compute T2 amplitudes and MP2 energy.
        wfn_MP2 = mp2_wfn(self.parameters, self.wfn)
        E_MP2, t2 = wfn_MP2.solve_MP2()

        # Setting initial variables for readability.
        C = self.C
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

        # Set up the gradient.
        Gradient_RHF = np.zeros((natom, 3))
        Gradient = np.zeros((natom, 3)) 

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3)) 

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3), dtype='complex128')

        # Set up derivative t-amplitude matrices.
        dT2_dR = []
        dT2_dH = []

        # Set up U-coefficient matrices for AAT calculations.
        U_R = []
        U_H = []

        # Set up derivative of normalization factors.
        N_R = []
        #N_H = [] # Magnetic field normalization factors are zero mathematically.

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))
    
        # First derivative matrices.
        h_a = []
        S_a = []
        F_a = []
        ERI_a = []
        U_a = []
        half_S = []

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_d1 = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_d1 = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_d1 = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_d1 = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            # Compute the half derivative overlap for AAT calculation.
            half_S_d1 = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_d1[a] = T_d1[a].np
                V_d1[a] = V_d1[a].np
                S_d1[a] = S_d1[a].np

                ERI_d1[a] = ERI_d1[a].np
                ERI_d1[a] = ERI_d1[a].swapaxes(1,2)
                half_S_d1[a] = half_S_d1[a].np

                # Computing skeleton (core) first derivative integrals.
                h_d1 = T_d1[a] + V_d1[a]
                F_d1 = T_d1[a] + V_d1[a] + np.einsum('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + np.einsum('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * np.einsum('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = F_d1[o,o].copy().astype('complex128') - np.einsum('ij,jj->ij', S_d1[a][o,o], F[o,o]) + np.einsum('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = F_d1[v,v].copy().astype('complex128') - np.einsum('ab,bb->ab', S_d1[a][v,v], F[v,v]) + np.einsum('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * np.einsum('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                U_d1[v,v] += B/D

                for j in range(no):
                    U_d1[j,j] = -0.5 * S_d1[a][j,j]
                for b in range(no,nbf):
                    U_d1[b,b] = -0.5 * S_d1[a][b,b]

                #print("CPHF Coefficients for Nuclear Perturbations:")
                #print(U_d1, "\n")

                # Appending to lists.
                S_a.append(S_d1[a])
                ERI_a.append(ERI_d1[a])
                h_a.append(h_d1)
                F_a.append(F_d1)
                U_a.append(U_d1)
                half_S.append(half_S_d1[a])

                # Computing the gradient of the Fock matrix.
                occ_eps = self.wfn.eps[o].reshape(-1,1) - self.wfn.eps[o]
                vir_eps = self.wfn.eps[v].reshape(-1,1) - self.wfn.eps[v]

                f_grad = np.zeros((nbf,nbf), dtype='complex128')

                f_grad[o,o] += F_d1[o,o].copy().astype('complex128')
                f_grad[o,o] += U_d1[o,o] * occ_eps
                f_grad[o,o] -= np.einsum('ij,jj->ij', S_d1[a][o,o], F[o,o])
                f_grad[o,o] += np.einsum('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o])
                f_grad[o,o] -= 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                for i in range(no):
                    f_grad[i,i] = F_d1[i,i]
                    f_grad[i,i] -= S_d1[a][i,i] * F[i,i]
                    f_grad[i,i] += np.einsum('em,em->', U_d1[v,o], 2*ERI[i,v,i,o] - ERI.swapaxes(2,3)[i,v,i,o])
                    f_grad[i,i] += np.einsum('em,me->', U_d1[v,o], 2*ERI[i,o,i,v] - ERI.swapaxes(2,3)[i,o,i,v])
                    f_grad[i,i] -= 0.5 * np.einsum('mn,mn->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])
                    f_grad[i,i] -= 0.5 * np.einsum('mn,nm->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])

                f_grad[v,v] += F_d1[v,v].copy().astype('complex128')
                f_grad[v,v] += U_d1[v,v] * vir_eps
                f_grad[v,v] -= np.einsum('ab,bb->ab', S_d1[a][v,v], F[v,v])
                f_grad[v,v] += np.einsum('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o])
                f_grad[v,v] -= 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])

                for b in range(nv):
                    b += no
                    f_grad[b,b] = F_d1[b,b]
                    f_grad[b,b] -= S_d1[a][b,b] * F[b,b]
                    f_grad[b,b] += np.einsum('em,em->', U_d1[v,o], 2*ERI[b,v,b,o] - ERI.swapaxes(2,3)[b,v,b,o])
                    f_grad[b,b] += np.einsum('em,me->', U_d1[v,o], 2*ERI[b,o,b,v] - ERI.swapaxes(2,3)[b,o,b,v])
                    f_grad[b,b] -= 0.5 * np.einsum('mn,mn->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])
                    f_grad[b,b] -= 0.5 * np.einsum('mn,nm->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])

                #print("Fock Matrix Derivative:")
                #print(f_grad, "\n")

                # Computing the Hartree-Fock contribution to the gradient.
                Gradient_RHF[N1][a] = 2 * np.einsum('ii->', h_d1[o,o])
                Gradient_RHF[N1][a] += np.einsum('ijij->', 2 * ERI_d1[a][o,o,o,o] - ERI_d1[a].swapaxes(2,3)[o,o,o,o])
                Gradient_RHF[N1][a] -= 2 * np.einsum('ii,ii->', S_d1[a][o,o], F[o,o])

                # Computing the gradient of the ERIs.
                ERI_grad = ERI_d1[a][o,o,v,v].astype('complex128')
                ERI_grad += np.einsum('pi,pjab->ijab', U_d1[:,o], ERI[:,o,v,v])
                ERI_grad += np.einsum('pj,ipab->ijab', U_d1[:,o], ERI[o,:,v,v])
                ERI_grad += np.einsum('pa,ijpb->ijab', U_d1[:,v], ERI[o,o,:,v])
                ERI_grad += np.einsum('pb,ijap->ijab', U_d1[:,v], ERI[o,o,v,:])
                #print("ERI_a", ERI_grad, "\n")

                # Computing energy gradient without derivative t-amplitudes.
                #E_grad = np.einsum('ijab,ijab->', 2*(2*t2-t2.swapaxes(2,3)), ERI_grad).astype('complex128')
                #E_grad -= np.einsum('ijab,kjab,ik->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[o,o])
                #E_grad -= np.einsum('ijab,ikab,jk->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[o,o])
                #E_grad += np.einsum('ijab,ijcb,ac->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[v,v])
                #E_grad += np.einsum('ijab,ijac,bc->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[v,v])
                #print("Energy Gradient:", E_grad + rhf_gradient[N1][a])

                # Computing t-amplitude derivatives.
                t2_grad = ERI_grad.copy()
                t2_grad -= np.einsum('kjab,ik->ijab', t2, f_grad[o,o])
                t2_grad -= np.einsum('ikab,kj->ijab', t2, f_grad[o,o])
                t2_grad += np.einsum('ijcb,ac->ijab', t2, f_grad[v,v])
                t2_grad += np.einsum('ijac,cb->ijab', t2, f_grad[v,v])
                t2_grad /= (wfn_MP2.D_ijab)

                # Compute derivative of the normalization factor.
                #N_a = - (1 / np.sqrt((1 + np.einsum('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))**3))
                #N_a *= np.einsum('ijab,ijab', t2_grad, 2*t2 - t2.swapaxes(2,3))
                N_a = - (1 / np.sqrt((1 + np.einsum('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_a *= 0.5 * (np.einsum('ijab,ijab', np.conjugate(t2_grad), 2*t2 - t2.swapaxes(2,3)) + np.einsum('ijab,ijab', t2_grad, np.conjugate(2*t2 - t2.swapaxes(2,3))))
                N_R.append(N_a)

                #print("Nuclear Displacement Derivative t-amplitudes:")
                #print("t2 grad", t2_grad, "\n")
                dT2_dR.append(t2_grad)
                U_R.append(U_d1)

                # Computing energy gradient with derivative t-amplitudes.
                E_grad = np.einsum('ijab,ijab->', t2_grad, (2*ERI[o,o,v,v]-ERI.swapaxes(2,3)[o,o,v,v])).astype('complex128')
                E_grad += np.einsum('ijab,ijab->', t2, (2*ERI_grad-ERI_grad.swapaxes(2,3)))
                #print("Energy Gradient:", E_grad1 + rhf_gradient[N1][a], "\n")

                Gradient[N1][a] += E_grad

        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np
        Gradient_RHF += Nuc_Gradient
        #print("RHF Gradient:")
        #print(Gradient_RHF)

        Gradient += Gradient_RHF
        #print("MP2 Gradient:")
        #print(Gradient)

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_mag = A_mag.swapaxes(1,2)
        G_mag = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        for a in range(3):
            mu_mag = np.einsum('mp,mn,nq->pq', np.conjugate(C), self.H.mu_mag[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_d1 = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field. Using negative sign to cancel that in Hamiltonian.
            B = h_d1[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
            U_d1[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_d1[o,v] += U_d1[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
            B = - h_d1[o,o].copy().astype('complex128') + np.einsum('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
            U_d1[o,o] += B/D

            D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
            B = - h_d1[v,v].copy().astype('complex128') + np.einsum('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
            U_d1[v,v] += B/D 

            #print("CPHF Coefficients for Magnetic Field Perturbations:")
            #print(U_d1,"\n")

            #for j in range(no):
            #    U_d1[j,j] = 0
            #for b in range(no,nbf):
            #    U_d1[b,b] = 0

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            occ_eps = self.wfn.eps[o].reshape(-1,1) - self.wfn.eps[o]
            vir_eps = self.wfn.eps[v].reshape(-1,1) - self.wfn.eps[v]

            f_grad = np.zeros((nbf,nbf), dtype='complex128')

            f_grad[o,o] -= h_d1[o,o].copy().astype('complex128')  # Sign change
            f_grad[o,o] += U_d1[o,o] * occ_eps
            f_grad[o,o] += np.einsum('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            for i in range(no):
                f_grad[i,i] = - h_d1[i,i] + np.einsum('em,em->', U_d1[v,o], A_mag.swapaxes(1,2)[i,v,i,o])  # Sign change

            f_grad[v,v] -= h_d1[v,v].copy().astype('complex128')  # Sign change
            f_grad[v,v] += U_d1[v,v] * vir_eps
            f_grad[v,v] += np.einsum('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])

            for b in range(nv):
                b += no
                f_grad[b,b] = - h_d1[b,b] + np.einsum('em,em->', U_d1[v,o], A_mag.swapaxes(1,2)[b,v,b,o])  # Sign change

            #print("Fock Matrix Derivative:")
            #print(f_grad, "\n")

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            ERI_grad = np.einsum('pi,abpj->abij', U_d1[:,o], ERI[v,v,:,o])
            ERI_grad += np.einsum('pj,abip->abij', U_d1[:,o], ERI[v,v,o,:])
            ERI_grad += np.einsum('pa,pbij->abij', np.conjugate(U_d1[:,v]), ERI[:,v,o,o])
            ERI_grad += np.einsum('pb,apij->abij', np.conjugate(U_d1[:,v]), ERI[v,:,o,o])
            #print("ERI_a", ERI_grad, "\n")

            # Computing t-amplitude derivatives with respect to a magnetic field.
            t2_grad = ERI_grad.copy().swapaxes(0,2).swapaxes(1,3)
            t2_grad -= np.einsum('kjab,ik->ijab', t2, f_grad[o,o])
            t2_grad -= np.einsum('ikab,kj->ijab', t2, f_grad[o,o])
            t2_grad += np.einsum('ijcb,ac->ijab', t2, f_grad[v,v])
            t2_grad += np.einsum('ijac,cb->ijab', t2, f_grad[v,v])
            t2_grad /= (wfn_MP2.D_ijab)

            # Compute derivative of the normalization factor. For a magnetic field, these are zero.
            #N_a = - (1 / np.sqrt((1 + np.einsum('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
            #N_a *= 0.5 * (np.einsum('ijab,ijab', np.conjugate(t2_grad), 2*t2 - t2.swapaxes(2,3)) + np.einsum('ijab,ijab', t2_grad, np.conjugate(2*t2 - t2.swapaxes(2,3))))
            #N_H.append(N_a)

            #print("Magnetic Field Derivative t-amplitudes:")
            #print("t2 grad", t2_grad, "\n")
            dT2_dH.append(t2_grad)
            U_H.append(U_d1)

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3), dtype='complex128')
        AAT_1 = np.zeros((natom * 3, 3), dtype='complex128')
        AAT_2 = np.zeros((natom * 3, 3), dtype='complex128')
        AAT_3 = np.zeros((natom * 3, 3), dtype='complex128')
        AAT_4 = np.zeros((natom * 3, 3), dtype='complex128')
        AAT_Norm = np.zeros((natom * 3, 3), dtype='complex128')

        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + np.einsum('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                # Computing the Hartree-Fock term of the AAT.
                AAT_HF[lambda_alpha][beta] += N**2 * 2 * np.einsum("em,em", U_H[beta][v, o], U_R[lambda_alpha][v, o] + half_S[lambda_alpha][o, v].T)

                # Computing first terms of the AATs.
                AAT_1[lambda_alpha][beta] += N**2 * np.einsum("ijab,ijab", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), dT2_dH[beta])

                # Computing the second term of the AATs.
                AAT_2[lambda_alpha][beta] += N**2 * 1.0 * np.einsum("ijab,ijab,kk", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][o, o]) 
                AAT_2[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("ijab,kjab,ki", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][o, o]) 
                AAT_2[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,ijcb,ac", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][v, v]) 

                # Computing the third term of the AATs.
                AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[lambda_alpha][o, o] + half_S[lambda_alpha][o, o].T)
                AAT_3[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[lambda_alpha][v, v] + half_S[lambda_alpha][v, v].T)

                # Computing the fourth term of the AATs.
                AAT_4[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o, o], U_R[lambda_alpha][o, o] + half_S[lambda_alpha][o, o].T)
                AAT_4[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v, v], U_R[lambda_alpha][v, v] + half_S[lambda_alpha][v, v].T)

                AAT_4[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v, o], U_R[lambda_alpha][v, o] + half_S[lambda_alpha][o, v].T)
                AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v, o], U_R[lambda_alpha][v, o] + half_S[lambda_alpha][o, v].T)
                AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v, o], U_R[lambda_alpha][v, o] + half_S[lambda_alpha][o, v].T)

                # Adding terms for full normalization.
                if normalization == 'full':
                    AAT_HF[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2.0 * np.einsum("nn",U_H[beta][o, o])
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 1.0 * np.einsum("ijab,ijab,kk", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o, o])  
                    AAT_Norm[lambda_alpha][beta] -= N * N_R[lambda_alpha] * 2.0 * np.einsum("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o, o])  
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2.0 * np.einsum("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v, v])
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 1.0 * np.einsum("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])


        print("Hartree-Fock AAT:")
        print(AAT_HF.imag, "\n")
        print("AAT Term 1:")
        print(AAT_1.imag, "\n")
        print("AAT Term 2:")
        print(AAT_2.imag, "\n")
        print("AAT Term 3:")
        print(AAT_3.imag, "\n")
        print("AAT Term 4:")
        print(AAT_4.imag, "\n")

        AAT = AAT_HF + AAT_1 + AAT_2 + AAT_3 + AAT_4 + AAT_Norm

        return AAT.imag






















