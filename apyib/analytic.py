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
        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)

        # Compute the electron repulsion integrals in the MO basis.
        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)

        # Set up the gradient.
        Gradient = np.zeros((natom, 3))

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Compute the perturbation-independent A matrix for the CPHF coefficients.
        A = 4 * ERI - ERI.swapaxes(1,2) - ERI.swapaxes(2,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ik,jl,ijkl->ijkl', np.eye(nbf), np.eye(nbf), F.reshape(1,nbf,1,nbf) - F.reshape(nbf,1,nbf,1)) + A
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
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B_d1 = -F_d1 + oe.contract('ij,j->ij', S_d1[a], eps) + oe.contract('kl,ikjl->ij', S_d1[a][o,o], 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

                # Solve for the CPHF coefficients.
                U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
                U_d1[o,o] -= 0.5 * S_d1[a][o,o]
                #U_d1[v,o] += oe.contract('ijkl,lk->ji', G_inv, B_d1[v,o])
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
                Gradient[N1][a] = 2 * oe.contract('ii->', h_d1[o,o])
                Gradient[N1][a] += oe.contract('ijij->', 2 * ERI_d1[a][o,o,o,o] - ERI_d1[a].swapaxes(2,3)[o,o,o,o])
                Gradient[N1][a] -= 2 * oe.contract('ii,i->', S_d1[a][o,o], eps[o])

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
                    F_ab.append(T_d2[ab] + V_d2[ab] + oe.contract('piqi->pq', 2 * ERI_d2[ab][:,o,:,o] - ERI_d2[ab].swapaxes(2,3)[:,o,:,o]))

        for N1 in atoms:
            for N2 in atoms:
                for a in range(3):
                    for b in range(3):
                        # Defining common index for the first and second derivative components with respect to the Hessian index.
                        alpha = 3*N1 + a
                        beta = 3*N2 + b
                        N_ab = N1*natom*9 + N2*9 + a*3 + b%3

                        # Computing the eta matrix.
                        eta_ab = oe.contract('im,jm->ij', U_a[alpha][o,:], U_a[beta][o,:]) + oe.contract('im,jm->ij', U_a[beta][o,:], U_a[alpha][o,:]) - oe.contract('im,jm->ij', S_a[alpha][o,:], S_a[beta][o,:]) - oe.contract('im,jm->ij', S_a[beta][o,:], S_a[alpha][o,:]) 

                        # Computing the Hessian.
                        Hessian[alpha][beta] += 2 * oe.contract('ii->', h_ab[N_ab][o,o]) + oe.contract('ijij->', 2 * ERI_ab[N_ab][o,o,o,o] - ERI_ab[N_ab].swapaxes(2,3)[o,o,o,o])
                        Hessian[alpha][beta] -= 2 * oe.contract('ii,i->', S_ab[N_ab][o,o], eps[o])
                        Hessian[alpha][beta] -= 2 * oe.contract('ii,i->', eta_ab, eps[o])
                        Hessian[alpha][beta] += 4 * oe.contract('mj,mj->', U_a[beta][:,o], F_a[alpha][:,o]) + 4 * oe.contract('mj,mj->', U_a[alpha][:,o], F_a[beta][:,o])
                        Hessian[alpha][beta] += 4 * oe.contract('mj,mj,m->', U_a[alpha][:,o], U_a[beta][:,o], eps)
                        Hessian[alpha][beta] += 4 * oe.contract('mj,nl,mjnl->', U_a[alpha][:,o], U_a[beta][:,o], A[:,o,:,o])

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
        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)

        # Compute the electron repulsion integrals in the MO basis.
        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

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
        G = oe.contract('ik,jl,ijkl->ijkl', np.eye(nbf), np.eye(nbf), F.reshape(1,nbf,1,nbf) - F.reshape(nbf,1,nbf,1)) + A 
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
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B_d1 = -F_d1 + oe.contract('ij,j->ij', S_d1[a], eps) + oe.contract('kl,ikjl->ij', S_d1[a][o,o], 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

                # Solve for the CPHF coefficients.
                U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
                U_d1[o,o] -= 0.5 * S_d1[a][o,o]
                #U_d1[v,o] += oe.contract('ijkl,lk->ji', G_inv, B_d1[v,o])
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
                f_grad += oe.contract('rp,rq->pq', U_d1, F) + oe.contract('rq,pr->pq', U_d1, F)
                ###f_grad += oe.contract('ri,prqi->pq', U_d1[:,o], 2*ERI[:,:,:,o] - ERI.swapaxes(2,3)[:,:,:,o])
                ###f_grad += oe.contract('ri,piqr->pq', U_d1[:,o], 2*ERI[:,o,:,:] - ERI.swapaxes(2,3)[:,o,:,:])
                f_grad += oe.contract('ri,prqi->pq', U_d1[:,o], A.swapaxes(1,2)[:,:,:,o])
                #print("Fock Matrix Derivative", f_grad, "\n")
                ERI_grad = ERI_d1[a][o,o,v,v].astype('complex128')
                ERI_grad += oe.contract('pi,pjab->ijab', U_d1[:,o], ERI[:,o,v,v])
                ERI_grad += oe.contract('pj,ipab->ijab', U_d1[:,o], ERI[o,:,v,v])
                ERI_grad += oe.contract('pa,ijpb->ijab', U_d1[:,v], ERI[o,o,:,v])
                ERI_grad += oe.contract('pb,ijap->ijab', U_d1[:,v], ERI[o,o,v,:])
                #print("ERI_a", ERI_grad, "\n") # This term matches Kirk's code.
                E_grad = oe.contract('ijab,ijab->', 2*(2*t2-t2.swapaxes(2,3)), ERI_grad).astype('complex128')
                E_grad -= oe.contract('ijab,kjab,ik->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[o,o])
                E_grad -= oe.contract('ijab,ikab,jk->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[o,o])
                E_grad += oe.contract('ijab,ijcb,ac->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[v,v])
                E_grad += oe.contract('ijab,ijac,bc->', t2, (2*t2-t2.swapaxes(2,3)), f_grad[v,v])
                #print("Energy Gradient:", E_grad + rhf_gradient[N1][a]) # This is giving the gradient minus the HF component. Correct!
                t2_grad = ERI_grad.copy()
                t2_grad -= oe.contract('kjab,ik->ijab', t2, f_grad[o,o])
                t2_grad -= oe.contract('ikab,kj->ijab', t2, f_grad[o,o])
                t2_grad += oe.contract('ijcb,ac->ijab', t2, f_grad[v,v])
                t2_grad += oe.contract('ijac,cb->ijab', t2, f_grad[v,v])
                t2_grad /= (wfn_MP2.D_ijab)
                #print("t2 grad", t2_grad, "\n") # This term matches Kirk's code.

                E_grad1 = oe.contract('ijab,ijab->', t2_grad, (2*ERI[o,o,v,v]-ERI.swapaxes(2,3)[o,o,v,v])).astype('complex128')
                E_grad1 += oe.contract('ijab,ijab->', t2, (2*ERI_grad-ERI_grad.swapaxes(2,3)))
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
        #            F_ab.append(T_d2[ab] + V_d2[ab] + oe.contract('piqi->pq', 2 * ERI_d2[ab][:,o,:,o] - ERI_d2[ab].swapaxes(2,3)[:,o,:,o]))

        #for N1 in atoms:
        #    for N2 in atoms:
        #        for a in range(3):
        #            for b in range(3):
        #                # Defining common index for the first and second derivative components with respect to the Hessian index.
        #                alpha = 3*N1 + a
        #                beta = 3*N2 + b
        #                N_ab = N1*natom*9 + N2*9 + a*3 + b%3

        #                # Computing the eta matrix.
        #                eta_ab = oe.contract('im,jm->ij', U_a[alpha][o,:], U_a[beta][o,:]) + oe.contract('im,jm->ij', U_a[beta][o,:], U_a[alpha][o,:]) - oe.contract('im,jm->ij', S_a[alpha][o,:], S_a[beta][o,:]) - oe.contract('im,jm->ij', S_a[beta][o,:], S_a[alpha][o,:])

        #                # Computing the Hessian.
        #                Hessian[alpha][beta] += 2 * oe.contract('ii->', h_ab[N_ab][o,o]) + oe.contract('ijij->', 2 * ERI_ab[N_ab][o,o,o,o] - ERI_ab[N_ab].swapaxes(2,3)[o,o,o,o])
        #                Hessian[alpha][beta] -= 2 * oe.contract('ii,i->', S_ab[N_ab][o,o], eps[o])
        #                Hessian[alpha][beta] -= 2 * oe.contract('ii,i->', eta_ab, eps[o])
        #                Hessian[alpha][beta] += 4 * oe.contract('mj,mj->', U_a[beta][:,o], F_a[alpha][:,o]) + 4 * oe.contract('mj,mj->', U_a[alpha][:,o], F_a[beta][:,o])
        #                Hessian[alpha][beta] += 4 * oe.contract('mj,mj,m->', U_a[alpha][:,o], U_a[beta][:,o], eps)
        #                Hessian[alpha][beta] += 4 * oe.contract('mj,nl,mjnl->', U_a[alpha][:,o], U_a[beta][:,o], A[:,o,:,o])

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
        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)

        # Compute the electron repulsion integrals in the MO basis.
        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)

        # Set up the gradient.
        Gradient = np.zeros((natom, 3)) 

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3)) 

        # Compute the perturbation-independent A matrix for the CPHF coefficients.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
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
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + oe.contract('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf), dtype='cdouble')
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = F_d1[o,o].copy().astype('complex128') - oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = F_d1[v,v].copy().astype('complex128') - oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
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
                f_grad[o,o] -= oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o])
                f_grad[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], (2*ERI[o,v,o,o] - ERI.swapaxes(2,3)[o,v,o,o]) + (2*ERI[o,o,o,v] - ERI.swapaxes(2,3)[o,o,o,v]).swapaxes(1,3))
                f_grad[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], (2*ERI[o,o,o,o] - ERI.swapaxes(2,3)[o,o,o,o]) + (2*ERI[o,o,o,o] - ERI.swapaxes(2,3)[o,o,o,o]).swapaxes(1,3))
                
                for i in range(no):
                    f_grad[i,i] = F_d1[i,i]
                    f_grad[i,i] -= S_d1[a][i,i] * F[i,i]
                    f_grad[i,i] += oe.contract('em,em->', U_d1[v,o], 2*ERI[i,v,i,o] - ERI.swapaxes(2,3)[i,v,i,o])
                    f_grad[i,i] += oe.contract('em,me->', U_d1[v,o], 2*ERI[i,o,i,v] - ERI.swapaxes(2,3)[i,o,i,v])
                    f_grad[i,i] -= 0.5 * oe.contract('mn,mn->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])
                    f_grad[i,i] -= 0.5 * oe.contract('mn,nm->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])

                f_grad[v,v] += F_d1[v,v].copy().astype('complex128')
                f_grad[v,v] += U_d1[v,v] * vir_eps
                f_grad[v,v] -= oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v])
                f_grad[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], (2*ERI[v,v,v,o] - ERI.swapaxes(2,3)[v,v,v,o]) + (2*ERI[v,o,v,v] - ERI.swapaxes(2,3)[v,o,v,v]).swapaxes(1,3))
                f_grad[v,v] -= 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], (2*ERI[v,o,v,o] - ERI.swapaxes(2,3)[v,o,v,o]) + (2*ERI[v,o,v,o] - ERI.swapaxes(2,3)[v,o,v,o]).swapaxes(1,3))
                    
                for b in range(nv):
                    b += no
                    f_grad[b,b] = F_d1[b,b]
                    f_grad[b,b] -= S_d1[a][b,b] * F[b,b]
                    f_grad[b,b] += oe.contract('em,em->', U_d1[v,o], 2*ERI[b,v,b,o] - ERI.swapaxes(2,3)[b,v,b,o])
                    f_grad[b,b] += oe.contract('em,me->', U_d1[v,o], 2*ERI[b,o,b,v] - ERI.swapaxes(2,3)[b,o,b,v])
                    f_grad[b,b] -= 0.5 * oe.contract('mn,mn->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])
                    f_grad[b,b] -= 0.5 * oe.contract('mn,nm->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])

                #print("Fock Matrix Derivative:")
                #print(f_grad, "\n")

                # Computing the gradient.
                Gradient[N1][a] = 2 * oe.contract('ii->', h_d1[o,o])
                Gradient[N1][a] += oe.contract('ijij->', 2 * ERI_d1[a][o,o,o,o] - ERI_d1[a].swapaxes(2,3)[o,o,o,o])
                Gradient[N1][a] -= 2 * oe.contract('ii,ii->', S_d1[a][o,o], F[o,o])

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
                    F_ab.append(T_d2[ab] + V_d2[ab] + oe.contract('piqi->pq', 2 * ERI_d2[ab][:,o,:,o] - ERI_d2[ab].swapaxes(2,3)[:,o,:,o]))

        for N1 in atoms:
            for N2 in atoms:
                for a in range(3):
                    for b in range(3):
                        # Defining common index for the first and second derivative components with respect to the Hessian index.
                        alpha = 3*N1 + a
                        beta = 3*N2 + b
                        N_ab = N1*natom*9 + N2*9 + a*3 + b%3

                        # Computing the eta matrix.
                        eta_ab = oe.contract('im,jm->ij', U_a[alpha][o,:], U_a[beta][o,:]) + oe.contract('im,jm->ij', U_a[beta][o,:], U_a[alpha][o,:]) - oe.contract('im,jm->ij', S_a[alpha][o,:], S_a[beta][o,:]) - oe.contract('im,jm->ij', S_a[beta][o,:], S_a[alpha][o,:])

                        # Computing the Hessian.
                        Hessian[alpha][beta] += 2 * oe.contract('ii->', h_ab[N_ab][o,o]) + oe.contract('ijij->', 2 * ERI_ab[N_ab][o,o,o,o] - ERI_ab[N_ab].swapaxes(2,3)[o,o,o,o])
                        Hessian[alpha][beta] -= 2 * oe.contract('ii,i->', S_ab[N_ab][o,o], eps[o])
                        Hessian[alpha][beta] -= 2 * oe.contract('ii,i->', eta_ab, eps[o])
                        Hessian[alpha][beta] += 4 * oe.contract('mj,mj->', U_a[beta][:,o], F_a[alpha][:,o]) + 4 * oe.contract('mj,mj->', U_a[alpha][:,o], F_a[beta][:,o])
                        Hessian[alpha][beta] += 4 * oe.contract('mj,mj,m->', U_a[alpha][:,o], U_a[beta][:,o], eps)
                        Hessian[alpha][beta] += 4 * oe.contract('mj,nl,mjnl->', U_a[alpha][:,o], U_a[beta][:,o], A[:,o,:,o])

        Nuc_Hessian = self.H.molecule.nuclear_repulsion_energy_deriv2().np
        Hessian += Nuc_Hessian
        print("Hessian:")
        print(Hessian)

        return Gradient, Hessian


    def compute_RHF_AATs_Canonical(self, orbitals='non-canonical'):
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
        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI) 
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI) 

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Set up U-coefficient matrices for AAT calculations.
        U_R = [] 
        U_H = [] 

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        # First derivative matrices.
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
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + oe.contract('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf))
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_d1[o,o].copy() - oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_d1[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_d1[v,v].copy() - oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_d1[v,v] += B/D

                    for j in range(no):
                        U_d1[j,j] = -0.5 * S_d1[a][j,j]
                    for b in range(no,nbf):
                        U_d1[b,b] = -0.5 * S_d1[a][b,b]

                if orbitals == 'non-canonical':
                    U_d1[f_,f_] = -0.5 * S_d1[a][f_,f_]
                    U_d1[o_,o_] = -0.5 * S_d1[a][o_,o_]
                    U_d1[v_,v_] = -0.5 * S_d1[a][v_,v_]

                # Appending to lists.
                half_S.append(half_S_d1[a])
                U_R.append(U_d1)

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_mag = A_mag.swapaxes(1,2)
        G_mag = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_d1 = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
            B = h_d1[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_d1 = np.zeros((nbf,nbf))
            U_d1[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_d1[o,v] += U_d1[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_d1[o,o].copy() + oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_d1[v,v].copy() + oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                U_d1[v,v] += B/D

                for j in range(no):
                    U_d1[j,j] = 0
                for b in range(no,nbf):
                    U_d1[b,b] = 0

            if orbitals == 'non-canonical':
                U_d1[f_,f_] = 0
                U_d1[o_,o_] = 0
                U_d1[v_,v_] = 0

            U_H.append(U_d1)

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3))

        # Compute AATs.
        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                # Computing the Hartree-Fock term of the AAT.
                AAT_HF[lambda_alpha][beta] += 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

        print("Hartree-Fock AAT:")
        print(AAT_HF, "\n")

        AAT = AAT_HF

        return AAT



    def compute_MP2_AATs_Canonical(self, normalization='full', orbitals='non-canonical'):
        # Compute T2 amplitudes and MP2 energy.
        wfn_MP2 = mp2_wfn(self.parameters, self.wfn)
        E_MP2, t2 = wfn_MP2.solve_MP2()

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
        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3)) 

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Set up derivative t-amplitude matrices.
        dT2_dR = []
        dT2_dH = []

        # Set up U-coefficient matrices for AAT calculations.
        U_R = []
        U_H = []

        # Set up derivative of normalization factors.
        N_R = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))
    
        # First derivative matrices.
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
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + oe.contract('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf))
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                ###### BEGIN NEW CODE ######

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_d1[o,o].copy() - oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_d1[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_d1[v,v].copy() - oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_d1[v,v] += B/D

                    for j in range(no):
                        U_d1[j,j] = -0.5 * S_d1[a][j,j]
                    for b in range(no,nbf):
                        U_d1[b,b] = -0.5 * S_d1[a][b,b]

                if orbitals == 'non-canonical':
                    U_d1[f_,f_] = -0.5 * S_d1[a][f_,f_]
                    U_d1[o_,o_] = -0.5 * S_d1[a][o_,o_]
                    U_d1[v_,v_] = -0.5 * S_d1[a][v_,v_]

                # Computing the gradient of the Fock matrix.
                df_dR = np.zeros((nbf,nbf))

                df_dR[o,o] += F_d1[o,o].copy()
                df_dR[o,o] += U_d1[o,o] * self.wfn.eps[o].reshape(-1,1) + U_d1[o,o].swapaxes(0,1) * self.wfn.eps[o]
                df_dR[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o])
                df_dR[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                df_dR[v,v] += F_d1[v,v].copy()
                df_dR[v,v] += U_d1[v,v] * self.wfn.eps[v].reshape(-1,1) + U_d1[v,v].swapaxes(0,1) * self.wfn.eps[v]
                df_dR[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o])
                df_dR[v,v] -= 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                #print("Nuclear Displacement Fock Matrix Derivative: New Code")
                #print(df_dR,"\n")

                # Computing the gradient of the ERIs.
                dERI_dR = ERI_d1[a].copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_d1[:,t], ERI[:,t,t,t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_d1[:,t], ERI[t,:,t,t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_d1[:,t], ERI[t,t,:,t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_d1[:,t], ERI[t,t,t,:])

                ###### END NEW CODE ######

                ## Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                #D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                #B = F_d1[o,o].copy() - oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                #U_d1[o,o] = 0
                #U_d1[o,o] += B/D

                #D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                #B = F_d1[v,v].copy() - oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                #U_d1[v,v] = 0
                #U_d1[v,v] += B/D

                #for j in range(no):
                #    U_d1[j,j] = -0.5 * S_d1[a][j,j]
                #for b in range(no,nbf):
                #    U_d1[b,b] = -0.5 * S_d1[a][b,b]

                ##print("CPHF Coefficients for Nuclear Perturbations:")
                ##print(U_d1, "\n")

                ## Appending to lists.
                #U_a.append(U_d1)
                #half_S.append(half_S_d1[a])

                ## Computing the gradient of the Fock matrix.
                #occ_eps = self.wfn.eps[o].reshape(-1,1) - self.wfn.eps[o]
                #vir_eps = self.wfn.eps[v].reshape(-1,1) - self.wfn.eps[v]

                #f_grad = np.zeros((nbf,nbf))

                #f_grad[o,o] += F_d1[o,o].copy()
                #f_grad[o,o] += U_d1[o,o] * occ_eps
                #f_grad[o,o] -= oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o])
                #f_grad[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o])
                #f_grad[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                #for i in range(no):
                #    f_grad[i,i] = F_d1[i,i]
                #    f_grad[i,i] -= S_d1[a][i,i] * F[i,i]
                #    f_grad[i,i] += oe.contract('em,em->', U_d1[v,o], 2*ERI[i,v,i,o] - ERI.swapaxes(2,3)[i,v,i,o])
                #    f_grad[i,i] += oe.contract('em,me->', U_d1[v,o], 2*ERI[i,o,i,v] - ERI.swapaxes(2,3)[i,o,i,v])
                #    f_grad[i,i] -= 0.5 * oe.contract('mn,mn->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])
                #    f_grad[i,i] -= 0.5 * oe.contract('mn,nm->', S_d1[a][o,o], 2*ERI[i,o,i,o] - ERI.swapaxes(2,3)[i,o,i,o])

                #f_grad[v,v] += F_d1[v,v].copy()
                #f_grad[v,v] += U_d1[v,v] * vir_eps
                #f_grad[v,v] -= oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v])
                #f_grad[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o])
                #f_grad[v,v] -= 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])

                #for b in range(nv):
                #    b += no
                #    f_grad[b,b] = F_d1[b,b]
                #    f_grad[b,b] -= S_d1[a][b,b] * F[b,b]
                #    f_grad[b,b] += oe.contract('em,em->', U_d1[v,o], 2*ERI[b,v,b,o] - ERI.swapaxes(2,3)[b,v,b,o])
                #    f_grad[b,b] += oe.contract('em,me->', U_d1[v,o], 2*ERI[b,o,b,v] - ERI.swapaxes(2,3)[b,o,b,v])
                #    f_grad[b,b] -= 0.5 * oe.contract('mn,mn->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])
                #    f_grad[b,b] -= 0.5 * oe.contract('mn,nm->', S_d1[a][o,o], 2*ERI[b,o,b,o] - ERI.swapaxes(2,3)[b,o,b,o])

                #print("Nuclear Displacement Fock Matrix Derivative: Old Code")
                #print(f_grad, "\n")

                ## Computing the gradient of the ERIs.
                #ERI_grad = ERI_d1[a][o_,o_,v_,v_]
                #ERI_grad += oe.contract('pi,pjab->ijab', U_d1[:,o_], ERI[:,o_,v_,v_])
                #ERI_grad += oe.contract('pj,ipab->ijab', U_d1[:,o_], ERI[o_,:,v_,v_])
                #ERI_grad += oe.contract('pa,ijpb->ijab', U_d1[:,v_], ERI[o_,o_,:,v_])
                #ERI_grad += oe.contract('pb,ijap->ijab', U_d1[:,v_], ERI[o_,o_,v_,:])

                # Computing t-amplitude derivatives.
                dt2_dR = dERI_dR.copy()[o_,o_,v_,v_]
                dt2_dR -= oe.contract('kjab,ik->ijab', t2, df_dR[o_,o_])
                dt2_dR -= oe.contract('ikab,kj->ijab', t2, df_dR[o_,o_])
                dt2_dR += oe.contract('ijcb,ac->ijab', t2, df_dR[v_,v_])
                dt2_dR += oe.contract('ijac,cb->ijab', t2, df_dR[v_,v_])
                dt2_dR /= (wfn_MP2.D_ijab)

                # Compute derivative of the normalization factor.
                #N_a = - (1 / np.sqrt((1 + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))**3))
                #N_a *= oe.contract('ijab,ijab', t2_grad, 2*t2 - t2.swapaxes(2,3))
                N_a = - (1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_a *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))
                N_R.append(N_a)

                #print("Nuclear Displacement Derivative t-amplitudes:")
                #print("t2 grad", t2_grad, "\n")
                dT2_dR.append(dt2_dR)
                U_R.append(U_d1)
                half_S.append(half_S_d1[a])

        # Delete excess variables.
        del dERI_dR; del dt2_dR; del df_dR; del T_d1; del V_d1; del S_d1; del ERI_d1; del half_S_d1; del h_d1; del F_d1; del B; del U_d1; del A; del G
        gc.collect()

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_mag = A_mag.swapaxes(1,2)
        G_mag = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_d1 = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
            B = h_d1[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_d1 = np.zeros((nbf,nbf))
            U_d1[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_d1[o,v] += U_d1[v,o].T

            ###### BEGIN NEW CODE ######

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_d1[o,o].copy() + oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_d1[v,v].copy() + oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                U_d1[v,v] += B/D

                for j in range(no):
                    U_d1[j,j] = 0
                for b in range(no,nbf):
                    U_d1[b,b] = 0

            if orbitals == 'non-canonical':
                U_d1[f_,f_] = 0
                U_d1[o_,o_] = 0
                U_d1[v_,v_] = 0

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            df_dH = np.zeros((nbf,nbf))

            df_dH[o,o] -= h_d1[o,o].copy()
            df_dH[o,o] += U_d1[o,o] * self.wfn.eps[o].reshape(-1,1) - U_d1[o,o].swapaxes(0,1) * self.wfn.eps[o]
            df_dH[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            df_dH[v,v] -= h_d1[v,v].copy()
            df_dH[v,v] += U_d1[v,v] * self.wfn.eps[v].reshape(-1,1) - U_d1[v,v].swapaxes(0,1) * self.wfn.eps[v]
            df_dH[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
            #print("Magnetic Field Fock Matrix Derivative: New Code")
            #print(df_dH, "\n")

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            dERI_dH = oe.contract('tr,pqts->pqrs', U_d1[:,t], ERI[t,t,:,t])
            dERI_dH += oe.contract('ts,pqrt->pqrs', U_d1[:,t], ERI[t,t,t,:])
            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_d1[:,t], ERI[:,t,t,t])
            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_d1[:,t], ERI[t,:,t,t])

            ###### END NEW CODE ######

            ## Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            #D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
            #B = - h_d1[o,o].copy() + oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
            #U_d1[o,o] = 0
            #U_d1[o,o] += B/D

            #D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
            #B = - h_d1[v,v].copy() + oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
            #U_d1[v,v] = 0
            #U_d1[v,v] += B/D

            #print("CPHF Coefficients for Magnetic Field Perturbations:")
            #print(U_d1,"\n")

            ##for j in range(no):
            ##    U_d1[j,j] = 0
            ##for b in range(no,nbf):
            ##    U_d1[b,b] = 0

            ## Computing the gradient of the Fock matrix with respect to a magnetic field.
            #occ_eps = self.wfn.eps[o].reshape(-1,1) - self.wfn.eps[o]
            #vir_eps = self.wfn.eps[v].reshape(-1,1) - self.wfn.eps[v]

            #f_grad = np.zeros((nbf,nbf))

            #f_grad[o,o] -= h_d1[o,o].copy()
            #f_grad[o,o] += U_d1[o,o] * occ_eps
            #f_grad[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            #for i in range(no):
            #    f_grad[i,i] = - h_d1[i,i] + oe.contract('em,em->', U_d1[v,o], A_mag.swapaxes(1,2)[i,v,i,o])

            #f_grad[v,v] -= h_d1[v,v].copy()
            #f_grad[v,v] += U_d1[v,v] * vir_eps
            #f_grad[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])

            #for b in range(nv):
            #    b += no
            #    f_grad[b,b] = - h_d1[b,b] + oe.contract('em,em->', U_d1[v,o], A_mag.swapaxes(1,2)[b,v,b,o])

            ##print("Magnetic Field Fock Matrix Derivative: Old Code")
            ##print(f_grad, "\n")

            ## Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            #ERI_grad = oe.contract('pi,abpj->abij', U_d1[:,o_], ERI[v_,v_,:,o_])
            #ERI_grad += oe.contract('pj,abip->abij', U_d1[:,o_], ERI[v_,v_,o_,:])
            #ERI_grad -= oe.contract('pa,pbij->abij', U_d1[:,v_], ERI[:,v_,o_,o_])
            #ERI_grad -= oe.contract('pb,apij->abij', U_d1[:,v_], ERI[v_,:,o_,o_])

            # Computing t-amplitude derivatives with respect to a magnetic field.
            dt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
            #dt2_dH -= oe.contract('kjab,ik->ijab', t2, df_dH[o_,o_])
            #dt2_dH -= oe.contract('ikab,kj->ijab', t2, df_dH[o_,o_])
            #dt2_dH += oe.contract('ijcb,ac->ijab', t2, df_dH[v_,v_])
            #dt2_dH += oe.contract('ijac,cb->ijab', t2, df_dH[v_,v_])
            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)

            dt2_dH /= (wfn_MP2.D_ijab)

            # Compute derivative of the normalization factor. For a magnetic field, these are zero.
            #N_a = - (1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
            #N_a *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(t2_grad), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', t2_grad, np.conjugate(2*t2 - t2.swapaxes(2,3))))
            #N_H.append(N_a)

            #print("Magnetic Field Derivative t-amplitudes:")
            #print("t2 grad", t2_grad, "\n")
            dT2_dH.append(dt2_dH)
            U_H.append(U_d1)

        # Delete excess variables.
        del dERI_dH; del dt2_dH; del df_dH; del h_d1; del B; del U_d1; del A_mag; del G_mag
        gc.collect()

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3))
        AAT_1 = np.zeros((natom * 3, 3))
        AAT_2 = np.zeros((natom * 3, 3))
        AAT_3 = np.zeros((natom * 3, 3))
        AAT_4 = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                # Computing the Hartree-Fock term of the AAT.
                AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

                # Computing first terms of the AATs.
                AAT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), dT2_dH[beta])

                # Computing the second term of the AATs.
                if orbitals == 'canonical':
                    #AAT_2[lambda_alpha][beta] += N**2 * 1.0 * oe.contract("ijab,ijab,kk", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                    AAT_2[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][o_, o_]) 
                    AAT_2[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][v_, v_]) 

                # Computing the third term of the AATs.
                AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[lambda_alpha][o_, o_] + half_S[lambda_alpha][o_, o_].T)
                AAT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[lambda_alpha][v_, v_] + half_S[lambda_alpha][v_, v_].T)

                # Computing the fourth term of the AATs.
                AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[lambda_alpha][o_, o] + half_S[lambda_alpha][o, o_].T) # Keep becasue [o_,o]
                if orbitals == 'canonical':
                    AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[lambda_alpha][v_, v_] + half_S[lambda_alpha][v_, v_].T)

                AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)
                AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[lambda_alpha][v_, o_] + half_S[lambda_alpha][o_, v_].T)
                AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

                # Adding terms for full normalization.
                if normalization == 'full':
                    if orbitals == 'canonical':
                        #AAT_HF[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2.0 * oe.contract("nn", U_H[beta][o, o]) # U_H[i,i] = 0
                        #AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 1.0 * oe.contract("ijab,ijab,kk", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                        AAT_Norm[lambda_alpha][beta] -= N * N_R[lambda_alpha] * 2.0 * oe.contract("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_])  
                        AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2.0 * oe.contract("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_])
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 1.0 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

        print("Hartree-Fock AAT:")
        print(AAT_HF, "\n")
        print("AAT Term 1:")
        print(AAT_1, "\n")
        print("AAT Term 2:")
        print(AAT_2, "\n")
        print("AAT Term 3:")
        print(AAT_3, "\n")
        print("AAT Term 4:")
        print(AAT_4, "\n")

        AAT = AAT_HF + AAT_1 + AAT_2 + AAT_3 + AAT_4 + AAT_Norm

        return AAT



    def compute_CISD_AATs_Canonical(self, normalization='full', orbitals='non-canonical'):
        # Compute T2 amplitudes and MP2 energy.
        wfn_CISD = ci_wfn(self.parameters, self.wfn)
        E_CISD, t1, t2 = wfn_CISD.solve_CISD()

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
        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)

        # Swap axes for Dirac notation.
        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>

        # Compute the Fock matrix in the MO basis.
        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Set up derivative t-amplitude matrices.
        dT1_dR = []
        dT1_dH = []
        dT2_dR = []
        dT2_dH = []

        # Set up U-coefficient matrices for AAT calculations.
        U_R = []
        U_H = []

        # Set up derivative of normalization factors.
        N_R = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        # First derivative matrices.
        U_a = []
        half_S = []

        # Compute OPD and TPD matrices for use in computing the energy gradient.
        # Compute normalize amplitudes.
        N = 1 / np.sqrt(1**2 + 2*oe.contract('ia,ia->', np.conjugate(t1), t1) + oe.contract('ijab,ijab->', np.conjugate(t2), 2*t2-t2.swapaxes(2,3)))
        t0_n = N.copy()
        t1_n = t1 * N
        t2_n = t2 * N

        # Build OPD.
        D_pq = np.zeros_like(F)
        D_pq[o_,o_] -= 2 * oe.contract('ja,ia->ij', np.conjugate(t1_n), t1_n) + 2 * oe.contract('jkab,ikab->ij', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[v_,v_] += 2 * oe.contract('ia,ib->ab', np.conjugate(t1_n), t1_n) + 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[o_,v_] += 2 * np.conjugate(t0_n) * t1_n + 2 * oe.contract('jb,ijab->ia', np.conjugate(t1_n), t2_n - t2_n.swapaxes(2,3))
        D_pq[v_,o_] += 2 * np.conjugate(t1_n.T) * t0_n + 2 * oe.contract('ijab,jb->ai', np.conjugate(t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pq = D_pq[t_,t_]

        # Build TPD.
        D_pqrs = np.zeros_like(ERI)
        D_pqrs[o_,o_,o_,o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[v_,v_,v_,v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[o_,v_,v_,o_] += 4 * oe.contract('ja,ib->iabj', np.conjugate(t1_n), t1_n)
        D_pqrs[o_,v_,o_,v_] -= 2 * oe.contract('ja,ib->iajb', np.conjugate(t1_n), t1_n)
        D_pqrs[v_,o_,o_,v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), 2*t2_n - t2_n.swapaxes(2,3))

        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)

        D_pqrs[o_,o_,v_,v_] += np.conjugate(t0_n) * (2*t2_n -t2_n.swapaxes(2,3))
        D_pqrs[v_,v_,o_,o_] += np.conjugate(2*t2_n.swapaxes(0,2).swapaxes(1,3) - t2_n.swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)) * t0_n
        D_pqrs[v_,o_,v_,v_] += 2 * oe.contract('ja,ijcb->aibc', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
        D_pqrs[o_,v_,o_,o_] -= 2 * oe.contract('kjab,ib->iajk', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pqrs[v_,v_,v_,o_] += 2 * oe.contract('jiab,jc->abci', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pqrs[o_,o_,o_,v_] -= 2 * oe.contract('kb,ijba->ijka', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
        D_pqrs = D_pqrs[t_,t_,t_,t_]

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
                F_d1 = T_d1[a] + V_d1[a] + oe.contract('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + oe.contract('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf))
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_d1[o,o].copy() - oe.contract('ij,jj->ij', S_d1[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_d1[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_d1[v,v].copy() - oe.contract('ab,bb->ab', S_d1[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_d1[v,v] += B/D

                    for j in range(no):
                        U_d1[j,j] = -0.5 * S_d1[a][j,j]
                    for b in range(no,nbf):
                        U_d1[b,b] = -0.5 * S_d1[a][b,b]

                if orbitals == 'non-canonical':
                    U_d1[f_,f_] = -0.5 * S_d1[a][f_,f_]
                    U_d1[o_,o_] = -0.5 * S_d1[a][o_,o_]
                    U_d1[v_,v_] = -0.5 * S_d1[a][v_,v_]

                # Appending to lists.
                U_a.append(U_d1)
                half_S.append(half_S_d1[a])

                # Computing the gradient of the Fock matrix.
                df_dR = np.zeros((nbf,nbf))

                df_dR[o,o] += F_d1[o,o].copy()
                df_dR[o,o] += U_d1[o,o] * self.wfn.eps[o].reshape(-1,1) + U_d1[o,o].swapaxes(0,1) * self.wfn.eps[o]
                df_dR[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o])
                df_dR[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                df_dR[v,v] += F_d1[v,v].copy()
                df_dR[v,v] += U_d1[v,v] * self.wfn.eps[v].reshape(-1,1) + U_d1[v,v].swapaxes(0,1) * self.wfn.eps[v]
                df_dR[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o])
                df_dR[v,v] -= 0.5 * oe.contract('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])

                # Computing the gradient of the ERIs.
                dERI_dR = ERI_d1[a].copy()
                dERI_dR += oe.contract('tp,tqrs->pqrs', U_d1[:,t], ERI[:,t,t,t])
                dERI_dR += oe.contract('tq,ptrs->pqrs', U_d1[:,t], ERI[t,:,t,t])
                dERI_dR += oe.contract('tr,pqts->pqrs', U_d1[:,t], ERI[t,t,:,t])
                dERI_dR += oe.contract('ts,pqrt->pqrs', U_d1[:,t], ERI[t,t,t,:])

                # Compute CISD energy gradient.
                dE_dR = oe.contract('pq,pq->', df_dR[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dR[t_,t_,t_,t_], D_pqrs)

                # Computing the HF energy gradient.
                dE_dR_HF = 2 * oe.contract('ii->', h_d1[o,o])
                dE_dR_HF += oe.contract('ijij->', 2 * ERI_d1[a][o,o,o,o] - ERI_d1[a].swapaxes(2,3)[o,o,o,o])
                dE_dR_HF -= 2 * oe.contract('ii,i->', S_d1[a][o,o], self.wfn.eps[o])
                dE_dR_HF += Nuc_Gradient[N1][a]

                dE_dR_tot = dE_dR + dE_dR_HF

                # Compute dT1_dR guess amplitudes.
                dt1_dR = -dE_dR * t1
                dt1_dR -= oe.contract('ji,ja->ia', df_dR[o_,o_], t1)
                dt1_dR += oe.contract('ab,ib->ia', df_dR[v_,v_], t1)
                dt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1) 
                dt1_dR += oe.contract('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                dt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
                dt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)
                dt1_dR /= wfn_CISD.D_ia

                # Compute dT2_dR guess amplitudes.
                dt2_dR = -dE_dR * t2
                dt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1) 
                dt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1) 
                dt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1) 
                dt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1) 
                dt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2) 
                dt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2) 
                dt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2) 
                dt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2) 
                dt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2) 
                dt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)    
                dt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2) 
                dt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
                dt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                dt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
                dt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR /= wfn_CISD.D_ijab

                # Solve for initial CISD energy gradient.
                dE_dR_proj =  2.0 * oe.contract('ia,ia->', t1, df_dR[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
                dt1_dR = dt1_dR.copy()
                dt2_dR = dt2_dR.copy()                

                # Start iterative procedure.
                iteration = 1
                while iteration <= self.parameters['max_iterations']:
                    dE_dR_proj_old = dE_dR_proj
                    dt1_dR_old = dt1_dR.copy()
                    dt2_dR_old = dt2_dR.copy()

                    # Solving for the derivative residuals.
                    dRt1_dR = df_dR.copy().swapaxes(0,1)[o_,v_]

                    dRt1_dR -= dE_dR_proj * t1
                    dRt1_dR -= oe.contract('ji,ja->ia', df_dR[o_,o_], t1)
                    dRt1_dR += oe.contract('ab,ib->ia', df_dR[v_,v_], t1)
                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1)
                    dRt1_dR += oe.contract('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)

                    dRt1_dR -= E_CISD * dt1_dR
                    dRt1_dR -= oe.contract('ji,ja->ia', F[o_,o_], dt1_dR)
                    dRt1_dR += oe.contract('ab,ib->ia', F[v_,v_], dt1_dR)
                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dR)
                    dRt1_dR += oe.contract('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dR - dt2_dR.swapaxes(2,3))
                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dR)
                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dR)

                    dRt2_dR = dERI_dR.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                    dRt2_dR -= dE_dR_proj * t2
                    dRt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1)
                    dRt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1)
                    dRt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1)
                    dRt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1)
                    dRt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2)
                    dRt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)

                    dRt2_dR -= E_CISD * dt2_dR
                    dRt2_dR += oe.contract('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dR)
                    dRt2_dR += oe.contract('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dR)
                    dRt2_dR -= oe.contract('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dR)
                    dRt2_dR -= oe.contract('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dR)
                    dRt2_dR += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dR)
                    dRt2_dR += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dR)
                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dR)

                    dt1_dR += dRt1_dR / wfn_CISD.D_ia
                    dt2_dR += dRt2_dR / wfn_CISD.D_ijab

                    # Compute new CISD energy gradient.
                    dE_dR_proj =  2.0 * oe.contract('ia,ia->', t1, df_dR[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                    dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                    # Compute new total energy gradient.
                    dE_dR_tot_proj = dE_dR_proj + dE_dR_HF

                    # Compute convergence data.
                    rms_dt1_dR = oe.contract('ia,ia->', dt1_dR_old - dt1_dR, dt1_dR_old - dt1_dR) 
                    rms_dt1_dR = np.sqrt(rms_dt1_dR)

                    rms_dt2_dR = oe.contract('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR) 
                    rms_dt2_dR = np.sqrt(rms_dt2_dR)
                    delta_dE_dR_proj = dE_dR_proj_old - dE_dR_proj

                    #if print_level > 0:
                    #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dR_proj, dE_dR_tot_proj, delta_dE_dR_proj, rms_dt1_dR, rms_dt2_dR))

                    if iteration > 1:
                        if abs(delta_dE_dR_proj) < self.parameters['e_convergence'] and rms_dt1_dR < self.parameters['d_convergence'] and rms_dt2_dR < self.parameters['d_convergence']:
                            #print("Convergence criteria met.")
                            break
                    if iteration == self.parameters['max_iterations']:
                        if abs(delta_dE_dR_proj) > self.parameters['e_convergence'] or rms_dt1_dR > self.parameters['d_convergence'] or rms_dt2_dR > self.parameters['d_convergence']:
                            print("Not converged.")
                    iteration += 1

                #print("Projected Nuclear Energy Gradient:", dE_dR_tot_proj)
                #print("Adjoint Nuclear Energy Gradient:", dE_dR_tot)
                #print("T1 Amplitudes Derivatives:")
                #print(dt1_dR,"\n")
                #print("T2 Amplitudes Derivatives:")
                #print(dt2_dR,"\n")

                # Compute derivative of the normalization factor.
                N_a = - (1 / np.sqrt((1 + 2*oe.contract('ia,ia', np.conjugate(t1), t1) + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_a *= 0.5 * (2*oe.contract('ia,ia', np.conjugate(dt1_dR), t1) + 2*oe.contract('ia,ia', dt1_dR, np.conjugate(t1)) + oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))
                N_R.append(N_a)

                dT1_dR.append(dt1_dR)
                dT2_dR.append(dt2_dR)
                U_R.append(U_d1)

        # Delete excess variables.
        del dERI_dR; del dt1_dR; del dt2_dR; del dRt1_dR; del dRt2_dR; del dt1_dR_old; del dt2_dR_old
        del df_dR; del T_d1; del V_d1; del S_d1; del ERI_d1; del half_S_d1; del h_d1; del F_d1; del B; del U_d1; del A; del G
        gc.collect()

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_mag = A_mag.swapaxes(1,2)
        G_mag = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_d1 = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
            B = h_d1[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_d1 = np.zeros((nbf,nbf))
            U_d1[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_d1[o,v] += U_d1[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_d1[o,o].copy() + oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_d1[v,v].copy() + oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                U_d1[v,v] += B/D

                for j in range(no):
                    U_d1[j,j] = 0
                for b in range(no,nbf):
                    U_d1[b,b] = 0

            if orbitals == 'non-canonical':
                U_d1[f_,f_] = 0
                U_d1[o_,o_] = 0
                U_d1[v_,v_] = 0

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            df_dH = np.zeros((nbf,nbf))

            df_dH[o,o] -= h_d1[o,o].copy()
            df_dH[o,o] += U_d1[o,o] * self.wfn.eps[o].reshape(-1,1) - U_d1[o,o].swapaxes(0,1) * self.wfn.eps[o]
            df_dH[o,o] += oe.contract('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            df_dH[v,v] -= h_d1[v,v].copy()
            df_dH[v,v] += U_d1[v,v] * self.wfn.eps[v].reshape(-1,1) - U_d1[v,v].swapaxes(0,1) * self.wfn.eps[v]
            df_dH[v,v] += oe.contract('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
            #print("Magnetic Field Fock Matrix Derivative: New Code")
            #print(df_dH, "\n")

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            dERI_dH = oe.contract('tr,pqts->pqrs', U_d1[:,t], ERI[t,t,:,t])
            dERI_dH += oe.contract('ts,pqrt->pqrs', U_d1[:,t], ERI[t,t,t,:])
            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_d1[:,t], ERI[:,t,t,t])
            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_d1[:,t], ERI[t,:,t,t])

            # Compute CISD energy gradient.
            dE_dH = oe.contract('pq,pq->', df_dH[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dH[t_,t_,t_,t_], D_pqrs)

            # Computing the HF energy gradient.
            dE_dH_HF = 2 * oe.contract('ii->', h_d1[o,o])
            dE_dH_tot = dE_dH + dE_dH_HF

            # Compute dT1_dR guess amplitudes.
            dt1_dH = -dE_dH * t1
            dt1_dH -= oe.contract('ji,ja->ia', df_dH[o_,o_], t1)
            dt1_dH += oe.contract('ab,ib->ia', df_dH[v_,v_], t1)
            dt1_dH += oe.contract('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
            dt1_dH += oe.contract('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
            dt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
            dt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)
            dt1_dH /= wfn_CISD.D_ia

            # Compute dT2_dR guess amplitudes.
            dt2_dH = -dE_dH * t2
            dt2_dH += oe.contract('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
            dt2_dH += oe.contract('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
            dt2_dH -= oe.contract('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
            dt2_dH -= oe.contract('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
            dt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
            dt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
            dt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH /= wfn_CISD.D_ijab

            # Solve for initial CISD energy gradient.
            dE_dH_proj =  2.0 * oe.contract('ia,ia->', t1, df_dH[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
            dE_dH_proj += 2.0 * oe.contract('ia,ia->', dt1_dH, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
            dt1_dH = dt1_dH.copy()
            dt2_dH = dt2_dH.copy()

            # Start iterative procedure.
            iteration = 1
            while iteration <= self.parameters['max_iterations']:
                dE_dH_proj_old = dE_dH_proj
                dt1_dH_old = dt1_dH.copy()
                dt2_dH_old = dt2_dH.copy()

                # Solving for the derivative residuals.
                dRt1_dH = df_dH.copy().swapaxes(0,1)[o_,v_]

                dRt1_dH -= dE_dH_proj * t1
                dRt1_dH -= oe.contract('ji,ja->ia', df_dH[o_,o_], t1)
                dRt1_dH += oe.contract('ab,ib->ia', df_dH[v_,v_], t1)
                dRt1_dH += oe.contract('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
                dRt1_dH += oe.contract('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                dRt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
                dRt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)

                dRt1_dH -= E_CISD * dt1_dH
                dRt1_dH -= oe.contract('ji,ja->ia', F[o_,o_], dt1_dH)
                dRt1_dH += oe.contract('ab,ib->ia', F[v_,v_], dt1_dH)
                dRt1_dH += oe.contract('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dH)
                dRt1_dH += oe.contract('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dH - dt2_dH.swapaxes(2,3))
                dRt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dH)
                dRt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dH)

                dRt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                dRt2_dH -= dE_dH_proj * t2
                dRt2_dH += oe.contract('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
                dRt2_dH += oe.contract('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
                dRt2_dH -= oe.contract('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
                dRt2_dH -= oe.contract('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
                dRt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
                dRt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
                dRt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
                dRt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
                dRt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
                dRt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
                dRt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
                dRt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)

                dRt2_dH -= E_CISD * dt2_dH
                dRt2_dH += oe.contract('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dH)
                dRt2_dH += oe.contract('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dH)
                dRt2_dH -= oe.contract('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dH)
                dRt2_dH -= oe.contract('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dH)
                dRt2_dH += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dH)
                dRt2_dH += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dH)
                dRt2_dH -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dH)
                dRt2_dH += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dH)
                dRt2_dH -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dH)
                dRt2_dH -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dH)

                dt1_dH += dRt1_dH / wfn_CISD.D_ia
                dt2_dH += dRt2_dH / wfn_CISD.D_ijab

                # Compute new CISD energy gradient.
                dE_dH_proj =  2.0 * oe.contract('ia,ia->', t1, df_dH[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dH_proj += 2.0 * oe.contract('ia,ia->', dt1_dH, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                # Compute new total energy gradient.
                dE_dH_tot_proj = dE_dH_proj + dE_dH_HF

                # Compute convergence data.
                rms_dt1_dH = oe.contract('ia,ia->', dt1_dH_old - dt1_dH, dt1_dH_old - dt1_dH)
                rms_dt1_dH = np.sqrt(rms_dt1_dH)

                rms_dt2_dH = oe.contract('ijab,ijab->', dt2_dH_old - dt2_dH, dt2_dH_old - dt2_dH)
                rms_dt2_dH = np.sqrt(rms_dt2_dH)
                delta_dE_dH_proj = dE_dH_proj_old - dE_dH_proj

                #if print_level > 0:
                #print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dH_proj, dE_dH_tot_proj, delta_dE_dH_proj, rms_dt1_dH, rms_dt2_dH))

                if iteration > 1:
                    if abs(delta_dE_dH_proj) < self.parameters['e_convergence'] and rms_dt1_dH < self.parameters['d_convergence'] and rms_dt2_dH < self.parameters['d_convergence']:
                        #print("Convergence criteria met.")
                        break
                if iteration == self.parameters['max_iterations']:
                    if abs(delta_dE_dH_proj) > self.parameters['e_convergence'] or rms_dt1_dH > self.parameters['d_convergence'] or rms_dt2_dH > self.parameters['d_convergence']:
                        print("Not converged.")
                iteration += 1

            #print("Projected Magnetic Field Energy Gradient:", dE_dH_tot_proj)
            #print("Adjoint Magnetic Field Energy Gradient:", dE_dH_tot)
            #print("T1 Amplitudes Derivatives:")
            #print(dt1_dH,"\n")
            #print("T2 Amplitudes Derivatives:")
            #print(dt2_dH,"\n")

            dT1_dH.append(dt1_dH)
            dT2_dH.append(dt2_dH)
            U_H.append(U_d1)

        # Delete excess variables.
        del dERI_dH; del dt1_dH; del dt2_dH; del dRt1_dH; del dRt2_dH; del dt1_dH_old; del dt2_dH_old
        del df_dH; del h_d1; del B; del U_d1; del A_mag; del G_mag
        gc.collect()

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3))
        AAT_S0 = np.zeros((natom * 3, 3))
        AAT_0S = np.zeros((natom * 3, 3))
        AAT_SS = np.zeros((natom * 3, 3))
        AAT_DS = np.zeros((natom * 3, 3))
        AAT_SD = np.zeros((natom * 3, 3))
        AAT_DD = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + 2*oe.contract('ia,ia', t1, t1) + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                # Computing the Hartree-Fock term of the AAT.
                AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

                # Singles/Refence terms.
                AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dT1_dR[lambda_alpha], U_H[beta][v_,o_])

                #AAT_S0[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,nn,ia", t1, U_H[beta][o,o], U_R[lambda_alpha][o_,v_] + half_S[lambda_alpha][v_,o_].T) # U_H[i,i] = 0
                AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T)
                AAT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], U_R[lambda_alpha][o_,o] + half_S[lambda_alpha][o,o_].T)

                # Reference/Singles terms.
                AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ck", dT1_dH[beta], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T)

                AAT_0S[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,nn,ck", t1, U_H[beta][o,o], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T) # U_H[i,i] = 0
                if orbitals == 'canonical':
                    AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fk", t1, U_H[beta][v_,v_], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T)
                AAT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,cn", t1, U_H[beta][o_,o], U_R[lambda_alpha][v_,o] + half_S[lambda_alpha][o,v_].T)                

                # Singles/Singles terms.
                AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dT1_dR[lambda_alpha], dT1_dH[beta])

                #AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,nn,kc", dT1_dR[lambda_alpha], U_H[beta][o,o], t1) # U_H[i,i] = 0
                if orbitals == 'canonical':
                    AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,cf,kf", dT1_dR[lambda_alpha], U_H[beta][v_,v_], t1)
                    AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,nk,nc", dT1_dR[lambda_alpha], U_H[beta][o_,o_], t1)

                AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, t1)
                AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], U_R[lambda_alpha][o_,o_] + half_S[lambda_alpha][o_,o_].T, t1)

                #AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,nn,ca,ka", t1, U_H[beta][o,o], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, t1)
                #AAT_SS[lambda_alpha][beta] -= N**2 * 4 * oe.contract("kc,nn,ik,ic", t1, U_H[beta][o,o], U_R[lambda_alpha][o_,o_] + half_S[lambda_alpha][o_,o_].T, t1)
                if orbitals == 'canonical':
                    AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fa,ka", t1, U_H[beta][v_,v_], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, t1)
                    AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fc,ik,if", t1, U_H[beta][v_,v_], U_R[lambda_alpha][o_,o_] + half_S[lambda_alpha][o_,o_].T, t1)
                    AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ca,na", t1, U_H[beta][o_,o_], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, t1)
                AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,kn,in,ic", t1, U_H[beta][o_,o], U_R[lambda_alpha][o_,o] + half_S[lambda_alpha][o,o_].T, t1)
                AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,kc,ia,ia", t1, U_H[beta][o_,v_], U_R[lambda_alpha][o_,v_] + half_S[lambda_alpha][v_,o_].T, t1)
                AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,fn,kc", t1, U_H[beta][v_,o], U_R[lambda_alpha][v_,o] + half_S[lambda_alpha][o,v_].T, t1)
                AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,fk,nc", t1, U_H[beta][v_,o_], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, t1)
                AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,cn,kf", t1, U_H[beta][v_,o], U_R[lambda_alpha][v_,o] + half_S[lambda_alpha][o,v_].T, t1)
                AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,ck,nf", t1, U_H[beta][v_,o_], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, t1)

                # Doubles/Singles terms.
                AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), U_H[beta][v_,o_], t1)

                AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ia,ikac", dT1_dH[beta], U_R[lambda_alpha][o_,v_] + half_S[lambda_alpha][v_,o_].T, 2*t2 - t2.swapaxes(2,3))

                if orbitals == 'canonical':
                    #AAT_DS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,nn,ia,ikac", t1, U_H[beta][o,o], U_R[lambda_alpha][o_,v_] + half_S[lambda_alpha][v_,o_].T, 2*t2 - t2.swapaxes(2,3)) # U_H[i,i] = 0
                    AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,ia,ikaf", t1, U_H[beta][v_,v_], U_R[lambda_alpha][o_,v_] + half_S[lambda_alpha][v_,o_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ia,inac", t1, U_H[beta][o_,o_], U_R[lambda_alpha][o_,v_] + half_S[lambda_alpha][v_,o_].T, 2*t2 - t2.swapaxes(2,3))
                AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,ik,incf", t1, U_H[beta][v_,o_], U_R[lambda_alpha][o_,o_] + half_S[lambda_alpha][o_,o_].T, 2*t2 - t2.swapaxes(2,3))
                AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,in,ikfc", t1, U_H[beta][v_,o], U_R[lambda_alpha][o_,o] + half_S[lambda_alpha][o,o_].T, 2*t2 - t2.swapaxes(2,3))
                AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,ca,knaf", t1, U_H[beta][v_,o_], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
                AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,fa,knca", t1, U_H[beta][v_,o_], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, 2*t2 - t2.swapaxes(2,3))

                # Singles/Doubles terms.
                AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dT1_dR[lambda_alpha], U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

                AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,dl,kc", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, t1)

                #AAT_SD[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,nn,em,imae", t1, U_H[beta][o,o], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # U_H[i,i] = 0
                AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ea,kice", t1, U_H[beta][o_,v_], U_R[lambda_alpha][v_,v_] + half_S[lambda_alpha][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
                AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,kc,im,kmca", t1, U_H[beta][o_,v_], U_R[lambda_alpha][o_,o_] + half_S[lambda_alpha][o_,o_].T, 2*t2 - t2.swapaxes(2,3))
                if orbitals == 'canonical':
                    AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ac,em,imce", t1, U_H[beta][v_,v_], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ec,em,imac", t1, U_H[beta][v_,v_], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,ki,em,kmae", t1, U_H[beta][o_,o_], U_R[lambda_alpha][v_,o_] + half_S[lambda_alpha][o_,v_].T, 2*t2 - t2.swapaxes(2,3))
                AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_H[beta][o_,o], U_R[lambda_alpha][v_,o] + half_S[lambda_alpha][o,v_].T, 2*t2 - t2.swapaxes(2,3))

                # Doubles/Doubles terms.
                AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), dT2_dH[beta])

                if orbitals == 'canonical':
                    AAT_DD[lambda_alpha][beta] += N**2 * 1 * oe.contract("ijab,ijab,kk", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                    AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][o_, o_]) 
                    AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dT2_dR[lambda_alpha] - dT2_dR[lambda_alpha].swapaxes(2,3), t2, U_H[beta][v_, v_]) 

                AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[lambda_alpha][o_, o_] + half_S[lambda_alpha][o_, o_].T)
                AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[lambda_alpha][v_, v_] + half_S[lambda_alpha][v_, v_].T)

                AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[lambda_alpha][o_, o] + half_S[lambda_alpha][o, o_].T)
                if orbitals == 'canonical':
                    AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[lambda_alpha][v_, v_] + half_S[lambda_alpha][v_, v_].T)
                AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)
                AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[lambda_alpha][v_, o_] + half_S[lambda_alpha][o_, v_].T)
                AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

                # Adding terms for full normalization. 
                if normalization == 'full':
                    #AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("nn", U_H[beta][o, o]) # U_H[i,i] = 0
                    if orbitals == 'canonical':
                        #AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 1 * oe.contract("ijab,ijab,kk", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                        AAT_Norm[lambda_alpha][beta] -= N * N_R[lambda_alpha] * 2 * oe.contract("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_])
                        AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_])
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("ia,ai", t1, U_H[beta][v_, o_])
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("kc,kc", t1, U_H[beta][o_, v_])
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
                    if orbitals == 'canonical':
                        #AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 4 * oe.contract("kc,nn,kc", t1, U_H[beta][o,o], t1) # U_H[i,i] = 0
                        AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("kc,cf,kf", t1, U_H[beta][v_,v_], t1)
                        AAT_Norm[lambda_alpha][beta] -= N * N_R[lambda_alpha] * 2 * oe.contract("kc,nk,nc", t1, U_H[beta][o_,o_], t1)
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)
                    AAT_Norm[lambda_alpha][beta] += N * N_R[lambda_alpha] * 2 * oe.contract("ia,kc,ikac", t1, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))


        print("Hartree-Fock AAT:")
        print(AAT_HF, "\n")
        print("Singles/Reference AAT:")
        print(AAT_S0, "\n")
        print("Reference/Singles AAT:")
        print(AAT_0S, "\n")
        print("Singles/Singles AAT:")
        print(AAT_SS, "\n")
        print("Doubles/Singles:")
        print(AAT_DS, "\n")
        print("Singles/Doubles:")
        print(AAT_SD, "\n")
        print("Doubles/Doubles:")
        print(AAT_DD, "\n")

        AAT = AAT_HF + AAT_S0 + AAT_0S + AAT_SS + AAT_DS + AAT_SD + AAT_DD + AAT_Norm

        return AAT
















