"""This script contains a set of functions for analytic evaluation of the Hessian."""

import numpy as np
import psi4
import gc
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



    def compute_RHF_AATs(self, orbitals='non-canonical'):
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
        G = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
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
                F_d1 = T_d1[a] + V_d1[a] + np.einsum('piqi->pq', 2 * ERI_d1[a][:,o,:,o] - ERI_d1[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_d1[v,o] + np.einsum('ai,ii->ai', S_d1[a][v,o], F[o,o]) + 0.5 * np.einsum('mn,amin->ai', S_d1[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_d1 = np.zeros((nbf,nbf))
                U_d1[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_d1[o,v] -= U_d1[v,o].T + S_d1[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_d1[o,o].copy() - np.einsum('ij,jj->ij', S_d1[a][o,o], F[o,o]) + np.einsum('em,iejm->ij', U_d1[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * np.einsum('mn,imjn->ij', S_d1[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_d1[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_d1[v,v].copy() - np.einsum('ab,bb->ab', S_d1[a][v,v], F[v,v]) + np.einsum('em,aebm->ab', U_d1[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * np.einsum('mn,ambn->ab', S_d1[a][o,o], A.swapaxes(1,2)[v,o,v,o])
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
        G_mag = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = np.einsum('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)

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
                B = - h_d1[o,o].copy() + np.einsum('em,iejm->ij', U_d1[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_d1[v,v].copy() + np.einsum('em,aebm->ab', U_d1[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
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
                AAT_HF[lambda_alpha][beta] += 2 * np.einsum("em,em", U_H[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)

        print("Hartree-Fock AAT:")
        print(AAT_HF, "\n")

        AAT = AAT_HF

        return AAT



    def compute_MP2_AATs(self, normalization='full', orbitals='non-canonical'):
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

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3))
        AAT_1 = np.zeros((natom * 3, 3))
        AAT_2 = np.zeros((natom * 3, 3))
        AAT_3 = np.zeros((natom * 3, 3))
        AAT_4 = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        # Compute normalization factor.
        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + np.einsum('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        # Setting up lists for magnetic field dependent terms.
        U_H = []
        dT2_dH = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_mag = A_mag.swapaxes(1,2)
        G_mag = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for b in range(3):
            mu_mag_AO[b] = -0.5 * mu_mag_AO[b].np
            mu_mag = np.einsum('mp,mn,nq->pq', C, mu_mag_AO[b], C)

            # Computing skeleton (core) first derivative integrals.
            h_core = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
            B = h_core[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_h = np.zeros((nbf,nbf))
            U_h[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_h[o,v] += U_h[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_core[o,o].copy() + np.einsum('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_h[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_core[v,v].copy() + np.einsum('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                U_h[v,v] += B/D

                for j in range(no):
                    U_h[j,j] = 0
                for c in range(no,nbf):
                    U_h[c,c] = 0

            if orbitals == 'non-canonical':
                U_h[f_,f_] = 0
                U_h[o_,o_] = 0
                U_h[v_,v_] = 0

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            df_dH = np.zeros((nbf,nbf))

            df_dH[o,o] -= h_core[o,o].copy()
            df_dH[o,o] += U_h[o,o] * self.wfn.eps[o].reshape(-1,1) - U_h[o,o].swapaxes(0,1) * self.wfn.eps[o]
            df_dH[o,o] += np.einsum('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            df_dH[v,v] -= h_core[v,v].copy()
            df_dH[v,v] += U_h[v,v] * self.wfn.eps[v].reshape(-1,1) - U_h[v,v].swapaxes(0,1) * self.wfn.eps[v]
            df_dH[v,v] += np.einsum('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            dERI_dH =  np.einsum('tr,pqts->pqrs', U_h[:,t], ERI[t,t,:,t])
            dERI_dH += np.einsum('ts,pqrt->pqrs', U_h[:,t], ERI[t,t,t,:])
            dERI_dH -= np.einsum('tp,tqrs->pqrs', U_h[:,t], ERI[:,t,t,t])
            dERI_dH -= np.einsum('tq,ptrs->pqrs', U_h[:,t], ERI[t,:,t,t])

            # Computing t-amplitude derivatives with respect to a magnetic field.
            dt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
            dt2_dH += np.einsum('ac,ijcb->ijab', df_dH[v_,v_], t2)
            dt2_dH += np.einsum('bc,ijac->ijab', df_dH[v_,v_], t2)
            dt2_dH -= np.einsum('ki,kjab->ijab', df_dH[o_,o_], t2)
            dt2_dH -= np.einsum('kj,ikab->ijab', df_dH[o_,o_], t2)
            dt2_dH /= (wfn_MP2.D_ijab)

            U_H.append(U_h)
            dT2_dH.append(dt2_dH)

        #del dt2_dH; del df_dH; del dERI_dH; del D; del B; del U_h; del A_mag; del G_mag
        #gc.collect()

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            # Compute the half derivative overlap for AAT calculation.
            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_core[a] = T_core[a].np
                V_core[a] = V_core[a].np
                S_core[a] = S_core[a].np

                ERI_core[a] = ERI_core[a].np
                ERI_core[a] = ERI_core[a].swapaxes(1,2)
                half_S_core[a] = half_S_core[a].np

                # Computing skeleton (core) first derivative integrals.
                h_core = T_core[a] + V_core[a]
                F_core = T_core[a] + V_core[a] + np.einsum('piqi->pq', 2 * ERI_core[a][:,o,:,o] - ERI_core[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_core[v,o] + np.einsum('ai,ii->ai', S_core[a][v,o], F[o,o]) + 0.5 * np.einsum('mn,amin->ai', S_core[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_R = np.zeros((nbf,nbf))
                U_R[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_R[o,v] -= U_R[v,o].T + S_core[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_core[o,o].copy() - np.einsum('ij,jj->ij', S_core[a][o,o], F[o,o]) + np.einsum('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * np.einsum('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_R[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_core[v,v].copy() - np.einsum('ab,bb->ab', S_core[a][v,v], F[v,v]) + np.einsum('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * np.einsum('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_R[v,v] += B/D

                    for j in range(no):
                        U_R[j,j] = -0.5 * S_core[a][j,j]
                    for b in range(no,nbf):
                        U_R[b,b] = -0.5 * S_core[a][b,b]

                if orbitals == 'non-canonical':
                    U_R[f_,f_] = -0.5 * S_core[a][f_,f_]
                    U_R[o_,o_] = -0.5 * S_core[a][o_,o_]
                    U_R[v_,v_] = -0.5 * S_core[a][v_,v_]

                # Computing the gradient of the Fock matrix.
                df_dR = np.zeros((nbf,nbf))

                df_dR[o,o] += F_core[o,o].copy()
                df_dR[o,o] += U_R[o,o] * self.wfn.eps[o].reshape(-1,1) + U_R[o,o].swapaxes(0,1) * self.wfn.eps[o]
                df_dR[o,o] += np.einsum('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o])
                df_dR[o,o] -= 0.5 * np.einsum('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                df_dR[v,v] += F_core[v,v].copy()
                df_dR[v,v] += U_R[v,v] * self.wfn.eps[v].reshape(-1,1) + U_R[v,v].swapaxes(0,1) * self.wfn.eps[v]
                df_dR[v,v] += np.einsum('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o])
                df_dR[v,v] -= 0.5 * np.einsum('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])

                # Computing the gradient of the ERIs.
                dERI_dR = ERI_core[a].copy()
                dERI_dR += np.einsum('tp,tqrs->pqrs', U_R[:,t], ERI[:,t,t,t])
                dERI_dR += np.einsum('tq,ptrs->pqrs', U_R[:,t], ERI[t,:,t,t])
                dERI_dR += np.einsum('tr,pqts->pqrs', U_R[:,t], ERI[t,t,:,t])
                dERI_dR += np.einsum('ts,pqrt->pqrs', U_R[:,t], ERI[t,t,t,:])

                # Computing t-amplitude derivatives.
                dt2_dR = dERI_dR.copy()[o_,o_,v_,v_]
                dt2_dR -= np.einsum('kjab,ik->ijab', t2, df_dR[o_,o_])
                dt2_dR -= np.einsum('ikab,kj->ijab', t2, df_dR[o_,o_])
                dt2_dR += np.einsum('ijcb,ac->ijab', t2, df_dR[v_,v_])
                dt2_dR += np.einsum('ijac,cb->ijab', t2, df_dR[v_,v_])
                dt2_dR /= (wfn_MP2.D_ijab)

                # Compute derivative of the normalization factor.
                N_R = - (1 / np.sqrt((1 + np.einsum('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_R *= 0.5 * (np.einsum('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + np.einsum('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))

                for beta in range(0,3):
                    #Setting up AAT indexing.
                    lambda_alpha = 3 * N1 + a

                    # Computing the Hartree-Fock term of the AAT.
                    AAT_HF[lambda_alpha][beta] += N**2 * 2 * np.einsum("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                    # Computing first terms of the AATs.
                    AAT_1[lambda_alpha][beta] += N**2 * np.einsum("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                    # Computing the second term of the AATs.
                    if orbitals == 'canonical':
                        #AAT_2[lambda_alpha][beta] += N**2 * 1.0 * np.einsum("ijab,ijab,kk", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                        AAT_2[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("ijab,kjab,ki", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o_, o_]) 
                        AAT_2[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,ijcb,ac", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][v_, v_]) 

                    # Computing the third term of the AATs.
                    AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
                    AAT_3[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)

                    # Computing the fourth term of the AATs.
                    AAT_4[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
                    if orbitals == 'canonical':
                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core[a][v_, v_].T)

                    AAT_4[lambda_alpha][beta] += N**2 * 2.0 * np.einsum("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
                    AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
                    AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * np.einsum("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                    # Adding terms for full normalization.
                    if normalization == 'full':
                        if orbitals == 'canonical':
                            #AAT_HF[lambda_alpha][beta] += N * N_R * 2.0 * np.einsum("nn", U_H[beta][o, o]) # U_H[i,i] = 0
                            #AAT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * np.einsum("ijab,ijab,kk", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2.0 * np.einsum("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_])  
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2.0 * np.einsum("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_])
                        AAT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * np.einsum("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

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



    def compute_CISD_AATs(self, normalization='full', orbitals='non-canonical'):
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
        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np

        # Set up the atomic axial tensor.
        AAT = np.zeros((natom * 3, 3))

        # Setting up different components of the AATs.
        AAT_HF = np.zeros((natom * 3, 3))
        AAT_S0 = np.zeros((natom * 3, 3))
        AAT_0S = np.zeros((natom * 3, 3))
        AAT_SS = np.zeros((natom * 3, 3))
        AAT_DS = np.zeros((natom * 3, 3))
        AAT_SD = np.zeros((natom * 3, 3))
        AAT_DD = np.zeros((natom * 3, 3))
        AAT_Norm = np.zeros((natom * 3, 3))

        # Compute normalization factor.
        if normalization == 'intermediate':
            N = 1
        elif normalization == 'full':
            N = 1 / np.sqrt(1 + 2*np.einsum('ia,ia', t1, t1) + np.einsum('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))

        # Set up derivative t-amplitude matrices.
        dT1_dH = []
        dT2_dH = []

        # Set up U-coefficient matrices for AAT calculations.
        U_H = []

        # Compute OPD and TPD matrices for use in computing the energy gradient.
        # Compute normalize amplitudes.
        N = 1 / np.sqrt(1**2 + 2*np.einsum('ia,ia->', np.conjugate(t1), t1) + np.einsum('ijab,ijab->', np.conjugate(t2), 2*t2-t2.swapaxes(2,3)))
        t0_n = N.copy()
        t1_n = t1 * N
        t2_n = t2 * N

        # Build OPD.
        D_pq = np.zeros_like(F)
        D_pq[o_,o_] -= 2 * np.einsum('ja,ia->ij', np.conjugate(t1_n), t1_n) + 2 * np.einsum('jkab,ikab->ij', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[v_,v_] += 2 * np.einsum('ia,ib->ab', np.conjugate(t1_n), t1_n) + 2 * np.einsum('ijac,ijbc->ab', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
        D_pq[o_,v_] += 2 * np.conjugate(t0_n) * t1_n + 2 * np.einsum('jb,ijab->ia', np.conjugate(t1_n), t2_n - t2_n.swapaxes(2,3))
        D_pq[v_,o_] += 2 * np.conjugate(t1_n.T) * t0_n + 2 * np.einsum('ijab,jb->ai', np.conjugate(t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pq = D_pq[t_,t_]

        # Build TPD.
        D_pqrs = np.zeros_like(ERI)
        D_pqrs[o_,o_,o_,o_] += np.einsum('klab,ijab->ijkl', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[v_,v_,v_,v_] += np.einsum('ijab,ijcd->abcd', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
        D_pqrs[o_,v_,v_,o_] += 4 * np.einsum('ja,ib->iabj', np.conjugate(t1_n), t1_n)
        D_pqrs[o_,v_,o_,v_] -= 2 * np.einsum('ja,ib->iajb', np.conjugate(t1_n), t1_n)
        D_pqrs[v_,o_,o_,v_] += 2 * np.einsum('jkac,ikbc->aijb', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), 2*t2_n - t2_n.swapaxes(2,3))

        D_pqrs[v_,o_,v_,o_] -= 4 * np.einsum('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * np.einsum('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] += 2 * np.einsum('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
        D_pqrs[v_,o_,v_,o_] -= 4 * np.einsum('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)

        D_pqrs[o_,o_,v_,v_] += np.conjugate(t0_n) * (2*t2_n -t2_n.swapaxes(2,3))
        D_pqrs[v_,v_,o_,o_] += np.conjugate(2*t2_n.swapaxes(0,2).swapaxes(1,3) - t2_n.swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)) * t0_n
        D_pqrs[v_,o_,v_,v_] += 2 * np.einsum('ja,ijcb->aibc', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
        D_pqrs[o_,v_,o_,o_] -= 2 * np.einsum('kjab,ib->iajk', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pqrs[v_,v_,v_,o_] += 2 * np.einsum('jiab,jc->abci', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
        D_pqrs[o_,o_,o_,v_] -= 2 * np.einsum('kb,ijba->ijka', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
        D_pqrs = D_pqrs[t_,t_,t_,t_]

        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_mag = A_mag.swapaxes(1,2)
        G_mag = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))

        # Get the magnetic dipole AO integrals and transform into the MO basis.
        mu_mag_AO = mints.ao_angular_momentum()
        for a in range(3):
            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
            mu_mag = np.einsum('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_core = mu_mag

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
            B = h_core[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            U_h = np.zeros((nbf,nbf))
            U_h[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
            U_h[o,v] += U_h[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_core[o,o].copy() + np.einsum('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
                U_h[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_core[v,v].copy() + np.einsum('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
                U_h[v,v] += B/D

                for j in range(no):
                    U_h[j,j] = 0
                for c in range(no,nbf):
                    U_h[c,c] = 0

            if orbitals == 'non-canonical':
                U_h[f_,f_] = 0
                U_h[o_,o_] = 0
                U_h[v_,v_] = 0

            # Computing the gradient of the Fock matrix with respect to a magnetic field.
            df_dH = np.zeros((nbf,nbf))

            df_dH[o,o] -= h_core[o,o].copy()
            df_dH[o,o] += U_h[o,o] * self.wfn.eps[o].reshape(-1,1) - U_h[o,o].swapaxes(0,1) * self.wfn.eps[o]
            df_dH[o,o] += np.einsum('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])

            df_dH[v,v] -= h_core[v,v].copy()
            df_dH[v,v] += U_h[v,v] * self.wfn.eps[v].reshape(-1,1) - U_h[v,v].swapaxes(0,1) * self.wfn.eps[v]
            df_dH[v,v] += np.einsum('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])

            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
            dERI_dH =  np.einsum('tr,pqts->pqrs', U_h[:,t], ERI[t,t,:,t])
            dERI_dH += np.einsum('ts,pqrt->pqrs', U_h[:,t], ERI[t,t,t,:])
            dERI_dH -= np.einsum('tp,tqrs->pqrs', U_h[:,t], ERI[:,t,t,t])
            dERI_dH -= np.einsum('tq,ptrs->pqrs', U_h[:,t], ERI[t,:,t,t])

            # Compute CISD energy gradient.
            dE_dH = np.einsum('pq,pq->', df_dH[t_,t_], D_pq) + np.einsum('pqrs,pqrs->', dERI_dH[t_,t_,t_,t_], D_pqrs)

            # Computing the HF energy gradient.
            dE_dH_HF = 2 * np.einsum('ii->', h_core[o,o])
            dE_dH_tot = dE_dH + dE_dH_HF

            # Compute dT1_dR guess amplitudes.
            dt1_dH = -dE_dH * t1 
            dt1_dH -= np.einsum('ji,ja->ia', df_dH[o_,o_], t1)
            dt1_dH += np.einsum('ab,ib->ia', df_dH[v_,v_], t1)
            dt1_dH += np.einsum('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
            dt1_dH += np.einsum('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
            dt1_dH += np.einsum('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
            dt1_dH -= np.einsum('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)
            dt1_dH /= wfn_CISD.D_ia

            # Compute dT2_dR guess amplitudes.
            dt2_dH = -dE_dH * t2 
            dt2_dH += np.einsum('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
            dt2_dH += np.einsum('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
            dt2_dH -= np.einsum('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
            dt2_dH -= np.einsum('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
            dt2_dH += np.einsum('ac,ijcb->ijab', df_dH[v_,v_], t2)
            dt2_dH += np.einsum('bc,ijac->ijab', df_dH[v_,v_], t2)
            dt2_dH -= np.einsum('ki,kjab->ijab', df_dH[o_,o_], t2)
            dt2_dH -= np.einsum('kj,ikab->ijab', df_dH[o_,o_], t2)
            dt2_dH += np.einsum('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
            dt2_dH += np.einsum('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
            dt2_dH -= np.einsum('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += np.einsum('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= np.einsum('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH -= np.einsum('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
            dt2_dH += np.einsum('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
            dt2_dH -= np.einsum('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)
            dt2_dH /= wfn_CISD.D_ijab

            # Solve for initial CISD energy gradient.
            dE_dH_proj =  2.0 * np.einsum('ia,ia->', t1, df_dH[o_,v_]) + np.einsum('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
            dE_dH_proj += 2.0 * np.einsum('ia,ia->', dt1_dH, F[o_,v_]) + np.einsum('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
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
                dRt1_dH -= np.einsum('ji,ja->ia', df_dH[o_,o_], t1)
                dRt1_dH += np.einsum('ab,ib->ia', df_dH[v_,v_], t1)
                dRt1_dH += np.einsum('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
                dRt1_dH += np.einsum('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                dRt1_dH += np.einsum('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
                dRt1_dH -= np.einsum('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)

                dRt1_dH -= E_CISD * dt1_dH
                dRt1_dH -= np.einsum('ji,ja->ia', F[o_,o_], dt1_dH)
                dRt1_dH += np.einsum('ab,ib->ia', F[v_,v_], dt1_dH)
                dRt1_dH += np.einsum('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dH)
                dRt1_dH += np.einsum('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dH - dt2_dH.swapaxes(2,3))
                dRt1_dH += np.einsum('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dH)
                dRt1_dH -= np.einsum('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dH)

                dRt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                dRt2_dH -= dE_dH_proj * t2
                dRt2_dH += np.einsum('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
                dRt2_dH += np.einsum('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
                dRt2_dH -= np.einsum('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
                dRt2_dH -= np.einsum('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
                dRt2_dH += np.einsum('ac,ijcb->ijab', df_dH[v_,v_], t2)
                dRt2_dH += np.einsum('bc,ijac->ijab', df_dH[v_,v_], t2)
                dRt2_dH -= np.einsum('ki,kjab->ijab', df_dH[o_,o_], t2)
                dRt2_dH -= np.einsum('kj,ikab->ijab', df_dH[o_,o_], t2)
                dRt2_dH += np.einsum('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
                dRt2_dH += np.einsum('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
                dRt2_dH -= np.einsum('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += np.einsum('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= np.einsum('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
                dRt2_dH -= np.einsum('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
                dRt2_dH += np.einsum('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
                dRt2_dH -= np.einsum('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)

                dRt2_dH -= E_CISD * dt2_dH
                dRt2_dH += np.einsum('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dH)
                dRt2_dH += np.einsum('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dH)
                dRt2_dH -= np.einsum('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dH)
                dRt2_dH -= np.einsum('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dH)
                dRt2_dH += np.einsum('ac,ijcb->ijab', F[v_,v_], dt2_dH)
                dRt2_dH += np.einsum('bc,ijac->ijab', F[v_,v_], dt2_dH)
                dRt2_dH -= np.einsum('ki,kjab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH -= np.einsum('kj,ikab->ijab', F[o_,o_], dt2_dH)
                dRt2_dH += np.einsum('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dH)
                dRt2_dH += np.einsum('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dH)
                dRt2_dH -= np.einsum('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += np.einsum('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= np.einsum('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dH)
                dRt2_dH -= np.einsum('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dH)
                dRt2_dH += np.einsum('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
                dRt2_dH -= np.einsum('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dH)

                dt1_dH += dRt1_dH / wfn_CISD.D_ia
                dt2_dH += dRt2_dH / wfn_CISD.D_ijab

                # Compute new CISD energy gradient.
                dE_dH_proj =  2.0 * np.einsum('ia,ia->', t1, df_dH[o_,v_]) + np.einsum('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dH_proj += 2.0 * np.einsum('ia,ia->', dt1_dH, F[o_,v_]) + np.einsum('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                # Compute new total energy gradient.
                dE_dH_tot_proj = dE_dH_proj + dE_dH_HF

                # Compute convergence data.
                rms_dt1_dH = np.einsum('ia,ia->', dt1_dH_old - dt1_dH, dt1_dH_old - dt1_dH)
                rms_dt1_dH = np.sqrt(rms_dt1_dH)

                rms_dt2_dH = np.einsum('ijab,ijab->', dt2_dH_old - dt2_dH, dt2_dH_old - dt2_dH)
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

            dT1_dH.append(dt1_dH)
            dT2_dH.append(dt2_dH)
            U_H.append(U_h)

        # Delete excess variables.
        #del dERI_dH; del dt1_dH; del dt2_dH; del dRt1_dH; del dRt2_dH; del dt1_dH_old; del dt2_dH_old
        #del df_dH; del h_core; del B; del U_h; del A_mag; del G_mag
        #gc.collect()


        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = np.einsum('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            # Compute the half derivative overlap for AAT calculation.
            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_core[a] = T_core[a].np
                V_core[a] = V_core[a].np
                S_core[a] = S_core[a].np

                ERI_core[a] = ERI_core[a].np
                ERI_core[a] = ERI_core[a].swapaxes(1,2)
                half_S_core[a] = half_S_core[a].np

                # Computing skeleton (core) first derivative integrals.
                h_core = T_core[a] + V_core[a]
                F_core = T_core[a] + V_core[a] + np.einsum('piqi->pq', 2 * ERI_core[a][:,o,:,o] - ERI_core[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_core[v,o] + np.einsum('ai,ii->ai', S_core[a][v,o], F[o,o]) + 0.5 * np.einsum('mn,amin->ai', S_core[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_R = np.zeros((nbf,nbf))
                U_R[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_R[o,v] -= U_R[v,o].T + S_core[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_core[o,o].copy() - np.einsum('ij,jj->ij', S_core[a][o,o], F[o,o]) + np.einsum('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * np.einsum('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_R[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_core[v,v].copy() - np.einsum('ab,bb->ab', S_core[a][v,v], F[v,v]) + np.einsum('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * np.einsum('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_R[v,v] += B/D

                    for j in range(no):
                        U_R[j,j] = -0.5 * S_core[a][j,j]
                    for c in range(no,nbf):
                        U_R[c,c] = -0.5 * S_core[a][c,c]

                if orbitals == 'non-canonical':
                    U_R[f_,f_] = -0.5 * S_core[a][f_,f_]
                    U_R[o_,o_] = -0.5 * S_core[a][o_,o_]
                    U_R[v_,v_] = -0.5 * S_core[a][v_,v_]

                # Computing the gradient of the Fock matrix.
                df_dR = np.zeros((nbf,nbf))

                df_dR[o,o] += F_core[o,o].copy()
                df_dR[o,o] += U_R[o,o] * self.wfn.eps[o].reshape(-1,1) + U_R[o,o].swapaxes(0,1) * self.wfn.eps[o]
                df_dR[o,o] += np.einsum('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o])
                df_dR[o,o] -= 0.5 * np.einsum('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])

                df_dR[v,v] += F_core[v,v].copy()
                df_dR[v,v] += U_R[v,v] * self.wfn.eps[v].reshape(-1,1) + U_R[v,v].swapaxes(0,1) * self.wfn.eps[v]
                df_dR[v,v] += np.einsum('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o])
                df_dR[v,v] -= 0.5 * np.einsum('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])

                # Computing the gradient of the ERIs.
                dERI_dR = ERI_core[a].copy()
                dERI_dR += np.einsum('tp,tqrs->pqrs', U_R[:,t], ERI[:,t,t,t])
                dERI_dR += np.einsum('tq,ptrs->pqrs', U_R[:,t], ERI[t,:,t,t])
                dERI_dR += np.einsum('tr,pqts->pqrs', U_R[:,t], ERI[t,t,:,t])
                dERI_dR += np.einsum('ts,pqrt->pqrs', U_R[:,t], ERI[t,t,t,:])

                # Compute CISD energy gradient.
                dE_dR = np.einsum('pq,pq->', df_dR[t_,t_], D_pq) + np.einsum('pqrs,pqrs->', dERI_dR[t_,t_,t_,t_], D_pqrs)

                # Computing the HF energy gradient.
                dE_dR_HF = 2 * np.einsum('ii->', h_core[o,o])
                dE_dR_HF += np.einsum('ijij->', 2 * ERI_core[a][o,o,o,o] - ERI_core[a].swapaxes(2,3)[o,o,o,o])
                dE_dR_HF -= 2 * np.einsum('ii,i->', S_core[a][o,o], self.wfn.eps[o])
                dE_dR_HF += Nuc_Gradient[N1][a]

                dE_dR_tot = dE_dR + dE_dR_HF

                # Compute dT1_dR guess amplitudes.
                dt1_dR = -dE_dR * t1
                dt1_dR -= np.einsum('ji,ja->ia', df_dR[o_,o_], t1)
                dt1_dR += np.einsum('ab,ib->ia', df_dR[v_,v_], t1)
                dt1_dR += np.einsum('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1) 
                dt1_dR += np.einsum('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                dt1_dR += np.einsum('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
                dt1_dR -= np.einsum('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)
                dt1_dR /= wfn_CISD.D_ia

                # Compute dT2_dR guess amplitudes.
                dt2_dR = -dE_dR * t2
                dt2_dR += np.einsum('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1) 
                dt2_dR += np.einsum('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1) 
                dt2_dR -= np.einsum('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1) 
                dt2_dR -= np.einsum('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1) 
                dt2_dR += np.einsum('ac,ijcb->ijab', df_dR[v_,v_], t2) 
                dt2_dR += np.einsum('bc,ijac->ijab', df_dR[v_,v_], t2) 
                dt2_dR -= np.einsum('ki,kjab->ijab', df_dR[o_,o_], t2) 
                dt2_dR -= np.einsum('kj,ikab->ijab', df_dR[o_,o_], t2) 
                dt2_dR += np.einsum('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2) 
                dt2_dR += np.einsum('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)    
                dt2_dR -= np.einsum('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2) 
                dt2_dR += np.einsum('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
                dt2_dR -= np.einsum('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR -= np.einsum('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                dt2_dR += np.einsum('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
                dt2_dR -= np.einsum('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)
                dt2_dR /= wfn_CISD.D_ijab

                # Solve for initial CISD energy gradient.
                dE_dR_proj =  2.0 * np.einsum('ia,ia->', t1, df_dR[o_,v_]) + np.einsum('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                dE_dR_proj += 2.0 * np.einsum('ia,ia->', dt1_dR, F[o_,v_]) + np.einsum('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
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
                    dRt1_dR -= np.einsum('ji,ja->ia', df_dR[o_,o_], t1)
                    dRt1_dR += np.einsum('ab,ib->ia', df_dR[v_,v_], t1)
                    dRt1_dR += np.einsum('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1)
                    dRt1_dR += np.einsum('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
                    dRt1_dR += np.einsum('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
                    dRt1_dR -= np.einsum('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)

                    dRt1_dR -= E_CISD * dt1_dR
                    dRt1_dR -= np.einsum('ji,ja->ia', F[o_,o_], dt1_dR)
                    dRt1_dR += np.einsum('ab,ib->ia', F[v_,v_], dt1_dR)
                    dRt1_dR += np.einsum('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dR)
                    dRt1_dR += np.einsum('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dR - dt2_dR.swapaxes(2,3))
                    dRt1_dR += np.einsum('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dR)
                    dRt1_dR -= np.einsum('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dR)

                    dRt2_dR = dERI_dR.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]

                    dRt2_dR -= dE_dR_proj * t2
                    dRt2_dR += np.einsum('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1)
                    dRt2_dR += np.einsum('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1)
                    dRt2_dR -= np.einsum('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1)
                    dRt2_dR -= np.einsum('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1)
                    dRt2_dR += np.einsum('ac,ijcb->ijab', df_dR[v_,v_], t2)
                    dRt2_dR += np.einsum('bc,ijac->ijab', df_dR[v_,v_], t2)
                    dRt2_dR -= np.einsum('ki,kjab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR -= np.einsum('kj,ikab->ijab', df_dR[o_,o_], t2)
                    dRt2_dR += np.einsum('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2)
                    dRt2_dR += np.einsum('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)
                    dRt2_dR -= np.einsum('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += np.einsum('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= np.einsum('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
                    dRt2_dR -= np.einsum('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
                    dRt2_dR += np.einsum('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
                    dRt2_dR -= np.einsum('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)

                    dRt2_dR -= E_CISD * dt2_dR
                    dRt2_dR += np.einsum('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dR)
                    dRt2_dR += np.einsum('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dR)
                    dRt2_dR -= np.einsum('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dR)
                    dRt2_dR -= np.einsum('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dR)
                    dRt2_dR += np.einsum('ac,ijcb->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR += np.einsum('bc,ijac->ijab', F[v_,v_], dt2_dR)
                    dRt2_dR -= np.einsum('ki,kjab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR -= np.einsum('kj,ikab->ijab', F[o_,o_], dt2_dR)
                    dRt2_dR += np.einsum('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dR)
                    dRt2_dR += np.einsum('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dR)
                    dRt2_dR -= np.einsum('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += np.einsum('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= np.einsum('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dR)
                    dRt2_dR -= np.einsum('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR += np.einsum('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
                    dRt2_dR -= np.einsum('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dR)

                    dt1_dR += dRt1_dR / wfn_CISD.D_ia
                    dt2_dR += dRt2_dR / wfn_CISD.D_ijab

                    # Compute new CISD energy gradient.
                    dE_dR_proj =  2.0 * np.einsum('ia,ia->', t1, df_dR[o_,v_]) + np.einsum('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
                    dE_dR_proj += 2.0 * np.einsum('ia,ia->', dt1_dR, F[o_,v_]) + np.einsum('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])

                    # Compute new total energy gradient.
                    dE_dR_tot_proj = dE_dR_proj + dE_dR_HF

                    # Compute convergence data.
                    rms_dt1_dR = np.einsum('ia,ia->', dt1_dR_old - dt1_dR, dt1_dR_old - dt1_dR) 
                    rms_dt1_dR = np.sqrt(rms_dt1_dR)

                    rms_dt2_dR = np.einsum('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR) 
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

                # Compute derivative of the normalization factor.
                N_R = - (1 / np.sqrt((1 + 2*np.einsum('ia,ia', np.conjugate(t1), t1) + np.einsum('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
                N_R *= 0.5 * (2*np.einsum('ia,ia', np.conjugate(dt1_dR), t1) + 2*np.einsum('ia,ia', dt1_dR, np.conjugate(t1)) + np.einsum('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + np.einsum('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))

                for beta in range(0,3):
                    #Setting up AAT indexing.
                    lambda_alpha = 3 * N1 + a

                    # Computing the Hartree-Fock term of the AAT.
                    AAT_HF[lambda_alpha][beta] += N**2 * 2 * np.einsum("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                    # Singles/Refence terms.
                    AAT_S0[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,ai", dt1_dR, U_H[beta][v_,o_])

                    #AAT_S0[lambda_alpha][beta] += N**2 * 4 * np.einsum("ia,nn,ia", t1, U_H[beta][o,o], U_R[o_,v_] + half_S_core[a][v_,o_].T) # U_H[i,i] = 0
                    AAT_S0[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,ei,ea", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T)
                    AAT_S0[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ia,am,im", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core[a][o,o_].T)

                    # Reference/Singles terms.
                    AAT_0S[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,ck", dT1_dH[beta], U_R[v_,o_] + half_S_core[a][o_,v_].T)

                    #AAT_0S[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,nn,ck", t1, U_H[beta][o,o], U_R[v_,o_] + half_S_core[a][o_,v_].T) # U_H[i,i] = 0
                    if orbitals == 'canonical':
                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,fc,fk", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core[a][o_,v_].T)
                    AAT_0S[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,kn,cn", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core[a][o,v_].T)                

                    # Singles/Singles terms.
                    AAT_SS[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,ia", dt1_dR, dT1_dH[beta])

                    #AAT_SS[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,nn,kc", dt1_dR, U_H[beta][o,o], t1) # U_H[i,i] = 0
                    if orbitals == 'canonical':
                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,cf,kf", dt1_dR, U_H[beta][v_,v_], t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,nk,nc", dt1_dR, U_H[beta][o_,o_], t1)

                    AAT_SS[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,ae,ie", dT1_dH[beta], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1)
                    AAT_SS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ia,mi,ma", dT1_dH[beta], U_R[o_,o_] + half_S_core[a][o_,o_].T, t1)

                    #AAT_SS[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,nn,ca,ka", t1, U_H[beta][o,o], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1)
                    #AAT_SS[lambda_alpha][beta] -= N**2 * 4 * np.einsum("kc,nn,ik,ic", t1, U_H[beta][o,o], U_R[o_,o_] + half_S_core[a][o_,o_].T, t1)
                    if orbitals == 'canonical':
                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,fc,fa,ka", t1, U_H[beta][v_,v_], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,fc,ik,if", t1, U_H[beta][v_,v_], U_R[o_,o_] + half_S_core[a][o_,o_].T, t1)
                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,kn,ca,na", t1, U_H[beta][o_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1)
                    AAT_SS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,kn,in,ic", t1, U_H[beta][o_,o], U_R[o_,o] + half_S_core[a][o,o_].T, t1)
                    AAT_SS[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,kc,ia,ia", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core[a][v_,o_].T, t1)
                    AAT_SS[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,fn,fn,kc", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core[a][o,v_].T, t1)
                    AAT_SS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,fn,fk,nc", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
                    AAT_SS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,fn,cn,kf", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core[a][o,v_].T, t1)
                    AAT_SS[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,fn,ck,nf", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)

                    # Doubles/Singles terms.
                    AAT_DS[lambda_alpha][beta] += N**2 * 2 * np.einsum("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)

                    AAT_DS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,ia,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3))

                    if orbitals == 'canonical':
                        #AAT_DS[lambda_alpha][beta] += N**2 * 4 * np.einsum("kc,nn,ia,ikac", t1, U_H[beta][o,o], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3)) # U_H[i,i] = 0
                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,fc,ia,ikaf", t1, U_H[beta][v_,v_], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3))
                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,kn,ia,inac", t1, U_H[beta][o_,o_], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_DS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,fn,ik,incf", t1, U_H[beta][v_,o_], U_R[o_,o_] + half_S_core[a][o_,o_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_DS[lambda_alpha][beta] -= N**2 * 2 * np.einsum("kc,fn,in,ikfc", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core[a][o,o_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_DS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,fn,ca,knaf", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_DS[lambda_alpha][beta] += N**2 * 2 * np.einsum("kc,fn,fa,knca", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))

                    # Singles/Doubles terms.
                    AAT_SD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

                    AAT_SD[lambda_alpha][beta] += N**2 * 2 * np.einsum("klcd,dl,kc", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)

                    #AAT_SD[lambda_alpha][beta] += N**2 * 4 * np.einsum("ia,nn,em,imae", t1, U_H[beta][o,o], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # U_H[i,i] = 0
                    AAT_SD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,kc,ea,kice", t1, U_H[beta][o_,v_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_SD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ia,kc,im,kmca", t1, U_H[beta][o_,v_], U_R[o_,o_] + half_S_core[a][o_,o_].T, 2*t2 - t2.swapaxes(2,3))
                    if orbitals == 'canonical':
                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,ac,em,imce", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3))
                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ia,ec,em,imac", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3))
                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ia,ki,em,kmae", t1, U_H[beta][o_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3))
                    AAT_SD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ia,km,em,kiea", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core[a][o,v_].T, 2*t2 - t2.swapaxes(2,3))

                    # Doubles/Doubles terms.
                    AAT_DD[lambda_alpha][beta] += N**2 * np.einsum("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

                    if orbitals == 'canonical':
                        #AAT_DD[lambda_alpha][beta] += N**2 * 1 * np.einsum("ijab,ijab,kk", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ijab,kjab,ki", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o_, o_]) 
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ijab,ijcb,ac", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][v_, v_]) 

                    AAT_DD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
                    AAT_DD[lambda_alpha][beta] += N**2 * 2 * np.einsum("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)

                    AAT_DD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
                    if orbitals == 'canonical':
                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core[a][v_, v_].T)
                    AAT_DD[lambda_alpha][beta] += N**2 * 2 * np.einsum("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
                    AAT_DD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
                    AAT_DD[lambda_alpha][beta] -= N**2 * 2 * np.einsum("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)

                    # Adding terms for full normalization. 
                    if normalization == 'full':
                        #AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("nn", U_H[beta][o, o]) # U_H[i,i] = 0
                        if orbitals == 'canonical':
                            #AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * np.einsum("ijab,ijab,kk", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o, o]) # U_H[i,i] = 0
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * np.einsum("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_])
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_])
                        AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * np.einsum("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

                        AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("ia,ai", t1, U_H[beta][v_, o_])
                        AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("kc,kc", t1, U_H[beta][o_, v_])
                        AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("ia,ia", t1, dT1_dH[beta])
                        if orbitals == 'canonical':
                            #AAT_Norm[lambda_alpha][beta] += N * N_R * 4 * np.einsum("kc,nn,kc", t1, U_H[beta][o,o], t1) # U_H[i,i] = 0
                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("kc,cf,kf", t1, U_H[beta][v_,v_], t1)
                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * np.einsum("kc,nk,nc", t1, U_H[beta][o_,o_], t1)
                        AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)
                        AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * np.einsum("ia,kc,ikac", t1, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))


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
















