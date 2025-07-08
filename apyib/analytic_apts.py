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



    def compute_RHF_APTs_LG(self, orbitals='non-canonical'):
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

        # Set up the atomic axial tensor.
        APT = np.zeros((natom * 3, 3))

        # Set up U-coefficient matrices for APT calculations.
        U_E = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        # Get the electric dipole AO integrals and transform into the MO basis.
        mu_AO = mints.ao_dipole()
        h_E = []
        F_E = []

        for b in range(3):
            mu_AO[b] = mu_AO[b].np
            mu = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_AO[b], C)

            # Computing skeleton (core) first derivative integrals.
            h_b = mu

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a electric field.
            B = -h_b[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a electric field.
            U_b = np.zeros((nbf,nbf))
            U_b[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
            U_b[o,v] -= U_b[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to an electric field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = h_b[o,o].copy() + oe.contract('em,iejm->ij', U_b[v,o], A.swapaxes(1,2)[o,v,o,o])
                U_b[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = h_b[v,v].copy() + oe.contract('em,aebm->ab', U_b[v,o], A.swapaxes(1,2)[v,v,v,o])
                U_b[v,v] += B/D

                for j in range(no):
                    U_b[j,j] = 0
                for c in range(no,nbf):
                    U_b[c,c] = 0

            if orbitals == 'non-canonical':
                U_b[f_,f_] = 0
                U_b[o_,o_] = 0
                U_b[v_,v_] = 0

            # Appending to lists.
            U_E.append(U_b)
            h_E.append(h_b)
            F_E.append(h_b)
            #print(U_b)

        for N1 in atoms:
            # Compute the nuclear skeleton (core) one-electron first derivative integrals in the MO basis.
            T_a = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_a = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_a = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the nuclear skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_a = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            # Compute the mixed nuclear/electric skeleton (core) one-electron second derivative integrals in the MO basis.
            h_ab = mints.ao_elec_dip_deriv1(N1)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_R = T_a[a].np
                V_R = V_a[a].np
                S_R = S_a[a].np

                ERI_R = ERI_a[a].np
                ERI_R = ERI_R.swapaxes(1,2)
                #A_R = (2 * ERI_R - ERI_R.swapaxes(2,3)) + (2 * ERI_R - ERI_R.swapaxes(2,3)).swapaxes(1,3)

                # Computing skeleton (core) first derivative integrals.
                h_R = T_R + V_R
                F_R = h_R + oe.contract('pkqk->pq', 2*ERI_R[:,o,:,o] - ERI_R[:,o,o,:].swapaxes(2,3))

                lambda_alpha = 3 * N1 + a
                for beta in range(3):
                    h_RE = h_ab[a + 3*beta].np
                    h_RE = oe.contract('mp,mn,nq->pq', np.conjugate(C), h_RE, C)

                    # Asymmetric approach to APT calculation.
                    APT[lambda_alpha][beta] += 2 * oe.contract('ii->', h_RE[o,o])
                    APT[lambda_alpha][beta] += 2 * oe.contract('pi,pi->', U_E[beta][:,o], F_R[:,o] + F_R[o,:].T) 
                    APT[lambda_alpha][beta] -= 2 * oe.contract('pi,pj,ij->', U_E[beta][:,o], S_R[:,o], F[o,o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('pj,ip,ij->', U_E[beta][:,o], S_R[o,:], F[o,o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,ij->', S_R[o,o], F_E[beta][o,o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,ki,kj->', S_R[o,o], U_E[beta][o,o], F[o,o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,kj,ik->', S_R[o,o], U_E[beta][o,o], F[o,o])
                    APT[lambda_alpha][beta] -= 2 * oe.contract('ij,pk,ipjk->', S_R[o,o],  U_E[beta][:,o], 2 * ERI[o,:,o,o] + 2 * ERI[o,o,o,:].swapaxes(1,3) - ERI[o,:,o,o].swapaxes(2,3) - ERI[o,o,:,o].swapaxes(1,2).swapaxes(2,3))

        # Compute the nuclear component of the APTs.
        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()

        N = np.zeros((3 * self.H.molecule.natom(), 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * self.H.molecule.natom()):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                N[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        APT = APT + N

        return APT



    def compute_RHF_APTs_VG(self, orbitals='non-canonical'):
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
        
        # Set up the atomic axial tensor.
        APT = np.zeros((natom * 3, 3))

        # Set up U-coefficient matrices for AAT calculations.
        U_R = [] 
        U_E = [] 

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
        A_elec = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A_elec = A_elec.swapaxes(1,2)
        G_elec = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_elec[v,o,v,o]
        G_elec = np.linalg.inv(G_elec.reshape((nv*no,nv*no)))

        # Get the electric dipole AO integrals and transform into the MO basis.
        mu_elec_AO = mints.ao_nabla()
        for a in range(3):
            mu_elec_AO[a] = - mu_elec_AO[a].np
            mu_elec = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_elec_AO[a], C)

            # Computing skeleton (core) first derivative integrals.
            h_d1 = mu_elec

            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to an electric field.
            B = h_d1[v,o]

            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to an electric field.
            U_d1 = np.zeros((nbf,nbf))
            U_d1[v,o] += (G_elec @ B.reshape((nv*no))).reshape(nv,no)
            U_d1[o,v] += U_d1[v,o].T

            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to an electric field.
            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                B = - h_d1[o,o].copy() + oe.contract('em,iejm->ij', U_d1[v,o], A_elec.swapaxes(1,2)[o,v,o,o])
                U_d1[o,o] += B/D

                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                B = - h_d1[v,v].copy() + oe.contract('em,aebm->ab', U_d1[v,o], A_elec.swapaxes(1,2)[v,v,v,o])
                U_d1[v,v] += B/D

                for j in range(no):
                    U_d1[j,j] = 0
                for b in range(no,nbf):
                    U_d1[b,b] = 0

            if orbitals == 'non-canonical':
                U_d1[f_,f_] = 0
                U_d1[o_,o_] = 0
                U_d1[v_,v_] = 0

            U_E.append(U_d1)
            #print(U_d1)

        # Setting up different components of the APTs.
        APT_HF = np.zeros((natom * 3, 3))

        # Compute APTs.
        for lambda_alpha in range(3 * natom):
            for beta in range(3):
                # Computing the Hartree-Fock term of the APT.
                APT_HF[lambda_alpha][beta] += 2 * oe.contract("em,em", U_E[beta][v_, o], U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)
                #print("Lambda Alpha:", lambda_alpha, "Beta:", beta)
                #print("U_R + half_S:")
                #print(U_R[lambda_alpha][v_, o] + half_S[lambda_alpha][o, v_].T)
                #print("U_E:")
                #print(U_E[beta][v_, o])


        # Compute the nuclear component of the APTs.
        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()

        N = np.zeros((3 * self.H.molecule.natom(), 3))
        delta_ab = np.eye(3)
        for lambd_alpha in range(3 * self.H.molecule.natom()):
            alpha = lambd_alpha % 3
            lambd = lambd_alpha // 3
            for beta in range(3):
                N[lambd_alpha][beta] += Z[lambd] * delta_ab[alpha, beta]

        APT = - 2 * APT_HF + N

        return APT



#    def compute_MP2_AATs(self, normalization='full', orbitals='non-canonical'):
#        # Compute T2 amplitudes and MP2 energy.
#        wfn_MP2 = mp2_wfn(self.parameters, self.wfn)
#        E_MP2, t2 = wfn_MP2.solve_MP2()
#
#        # Setting initial variables for readability.
#        C = self.C
#        nbf = self.wfn.nbf
#        no = self.wfn.ndocc
#        nv = self.wfn.nbf - self.wfn.ndocc
#
#        # Setting up slices.
#        C_list, I_list = get_slices(self.parameters, self.wfn)
#        f_ = C_list[0]
#        o_ = C_list[1]
#        v_ = C_list[2]
#        t_ = C_list[3]
#
#        o = slice(0, no) 
#        v = slice(no, nbf)
#        t = slice(0, nbf)
#
#        # Create a Psi4 matrix object for obtaining the perturbed MO basis integrals.
#        C_p4 = psi4.core.Matrix.from_array(C)
#    
#        # Set the atom lists for Hessian.
#        natom = self.H.molecule.natom()
#        atoms = np.arange(0, natom)
#
#        # Compute the core Hamiltonian in the MO basis.
#        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)
#
#        # Compute the electron repulsion integrals in the MO basis.
#        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
#        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
#        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
#        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)
#
#        # Swap axes for Dirac notation.
#        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>
#
#        # Compute the Fock matrix in the MO basis.
#        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])
#
#        # Use the MintsHelper to get the AO integrals from Psi4.
#        mints = psi4.core.MintsHelper(self.H.basis_set)
#
#        # Set up the atomic axial tensor.
#        AAT = np.zeros((natom * 3, 3))
#
#        # Setting up different components of the AATs.
#        AAT_HF = np.zeros((natom * 3, 3))
#        AAT_1 = np.zeros((natom * 3, 3))
#        AAT_2 = np.zeros((natom * 3, 3))
#        AAT_3 = np.zeros((natom * 3, 3))
#        AAT_4 = np.zeros((natom * 3, 3))
#        AAT_Norm = np.zeros((natom * 3, 3))
#
#        # Compute normalization factor.
#        if normalization == 'intermediate':
#            N = 1
#        elif normalization == 'full':
#            N = 1 / np.sqrt(1 + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))
#
#        # Setting up lists for magnetic field dependent terms.
#        U_H = []
#        dT2_dH = []
#
#        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
#        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
#        A_mag = A_mag.swapaxes(1,2)
#        G_mag = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
#        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))
#
#        # Get the magnetic dipole AO integrals and transform into the MO basis.
#        mu_mag_AO = mints.ao_angular_momentum()
#        for b in range(3):
#            mu_mag_AO[b] = -0.5 * mu_mag_AO[b].np
#            mu_mag = oe.contract('mp,mn,nq->pq', C, mu_mag_AO[b], C)
#
#            # Computing skeleton (core) first derivative integrals.
#            h_core = mu_mag
#
#            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
#            B = h_core[v,o]
#
#            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
#            U_h = np.zeros((nbf,nbf))
#            U_h[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
#            U_h[o,v] += U_h[v,o].T
#
#            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
#            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
#                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
#                B = - h_core[o,o].copy() + oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
#                U_h[o,o] += B/D
#
#                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
#                B = - h_core[v,v].copy() + oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
#                U_h[v,v] += B/D
#
#                for j in range(no):
#                    U_h[j,j] = 0
#                for c in range(no,nbf):
#                    U_h[c,c] = 0
#
#            if orbitals == 'non-canonical':
#                U_h[f_,f_] = 0
#                U_h[o_,o_] = 0
#                U_h[v_,v_] = 0
#
#            # Computing the gradient of the Fock matrix with respect to a magnetic field.
#            df_dH = np.zeros((nbf,nbf))
#
#            df_dH[o,o] -= h_core[o,o].copy()
#            df_dH[o,o] += U_h[o,o] * self.wfn.eps[o].reshape(-1,1) - U_h[o,o].swapaxes(0,1) * self.wfn.eps[o]
#            df_dH[o,o] += oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
#
#            df_dH[v,v] -= h_core[v,v].copy()
#            df_dH[v,v] += U_h[v,v] * self.wfn.eps[v].reshape(-1,1) - U_h[v,v].swapaxes(0,1) * self.wfn.eps[v]
#            df_dH[v,v] += oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
#
#            #print("Magnetic Field Perturbed Fock Matrix:")
#            #print(df_dH)
#
#            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
#            dERI_dH =  oe.contract('tr,pqts->pqrs', U_h[:,t], ERI[t,t,:,t])
#            dERI_dH += oe.contract('ts,pqrt->pqrs', U_h[:,t], ERI[t,t,t,:])
#            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_h[:,t], ERI[:,t,t,t])
#            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_h[:,t], ERI[t,:,t,t])
#
#            # Computing t-amplitude derivatives with respect to a magnetic field.
#            dt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
#            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
#            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
#            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
#            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
#            dt2_dH /= (wfn_MP2.D_ijab)
#
#            U_H.append(U_h)
#            dT2_dH.append(dt2_dH)
#
#        #del dt2_dH; del df_dH; del dERI_dH; del D; del B; del U_h; del A_mag; del G_mag
#        #gc.collect()
#
#        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
#        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
#        A = A.swapaxes(1,2)
#        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
#        G = np.linalg.inv(G.reshape((nv*no,nv*no)))
#
#        # Compute and store first derivative integrals.
#        for N1 in atoms:
#            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
#            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
#            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
#            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
#
#            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
#            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
#
#            # Compute the half derivative overlap for AAT calculation.
#            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)
#
#            for a in range(3):
#                # Convert the Psi4 matrices to numpy matrices.
#                T_core[a] = T_core[a].np
#                V_core[a] = V_core[a].np
#                S_core[a] = S_core[a].np
#
#                ERI_core[a] = ERI_core[a].np
#                ERI_core[a] = ERI_core[a].swapaxes(1,2)
#                half_S_core[a] = half_S_core[a].np
#
#                # Computing skeleton (core) first derivative integrals.
#                h_core = T_core[a] + V_core[a]
#                F_core = T_core[a] + V_core[a] + oe.contract('piqi->pq', 2 * ERI_core[a][:,o,:,o] - ERI_core[a].swapaxes(2,3)[:,o,:,o])
#
#                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
#                B = -F_core[v,o] + oe.contract('ai,ii->ai', S_core[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_core[a][o,o], A.swapaxes(1,2)[v,o,o,o])
#
#                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
#                U_R = np.zeros((nbf,nbf))
#                U_R[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
#                U_R[o,v] -= U_R[v,o].T + S_core[a][o,v]
#
#                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
#                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
#                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
#                    B = F_core[o,o].copy() - oe.contract('ij,jj->ij', S_core[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
#                    U_R[o,o] += B/D
#
#                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
#                    B = F_core[v,v].copy() - oe.contract('ab,bb->ab', S_core[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
#                    U_R[v,v] += B/D
#
#                    for j in range(no):
#                        U_R[j,j] = -0.5 * S_core[a][j,j]
#                    for b in range(no,nbf):
#                        U_R[b,b] = -0.5 * S_core[a][b,b]
#
#                if orbitals == 'non-canonical':
#                    U_R[f_,f_] = -0.5 * S_core[a][f_,f_]
#                    U_R[o_,o_] = -0.5 * S_core[a][o_,o_]
#                    U_R[v_,v_] = -0.5 * S_core[a][v_,v_]
#
#                # Computing the gradient of the Fock matrix.
#                df_dR = np.zeros((nbf,nbf))
#
#                df_dR[o,o] += F_core[o,o].copy()
#                df_dR[o,o] += U_R[o,o] * self.wfn.eps[o].reshape(-1,1) + U_R[o,o].swapaxes(0,1) * self.wfn.eps[o]
#                df_dR[o,o] += oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o])
#                df_dR[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
#
#                df_dR[v,v] += F_core[v,v].copy()
#                df_dR[v,v] += U_R[v,v] * self.wfn.eps[v].reshape(-1,1) + U_R[v,v].swapaxes(0,1) * self.wfn.eps[v]
#                df_dR[v,v] += oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o])
#                df_dR[v,v] -= 0.5 * oe.contract('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
#
#                #print("Nuclear Displaced Perturbed Fock Matrix:")
#                #print(df_dR)
#
#                # Computing the gradient of the ERIs.
#                dERI_dR = ERI_core[a].copy()
#                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:,t], ERI[:,t,t,t])
#                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:,t], ERI[t,:,t,t])
#                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:,t], ERI[t,t,:,t])
#                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:,t], ERI[t,t,t,:])
#
#                # Computing t-amplitude derivatives.
#                dt2_dR = dERI_dR.copy()[o_,o_,v_,v_]
#                dt2_dR -= oe.contract('kjab,ik->ijab', t2, df_dR[o_,o_])
#                dt2_dR -= oe.contract('ikab,kj->ijab', t2, df_dR[o_,o_])
#                dt2_dR += oe.contract('ijcb,ac->ijab', t2, df_dR[v_,v_])
#                dt2_dR += oe.contract('ijac,cb->ijab', t2, df_dR[v_,v_])
#                dt2_dR /= (wfn_MP2.D_ijab)
#
#                # Compute derivative of the normalization factor.
#                N_R = - (1 / np.sqrt((1 + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
#                N_R *= 0.5 * (oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))
#
#                for beta in range(0,3):
#                    #Setting up AAT indexing.
#                    lambda_alpha = 3 * N1 + a
#
#                    if orbitals == 'canonical':
#                        # Computing the Hartree-Fock term of the AAT.
#                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Computing first terms of the AATs.
#                        AAT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])
#
#                        # Computing the second term of the AATs.
#                        AAT_2[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
#                        AAT_2[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical
#
#                        # Computing the third term of the AATs.
#                        AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
#                        AAT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)
#
#                        # Computing the fourth term of the AATs.
#                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
#                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core[a][v_, v_].T) # Canonical
#
#                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
#                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Adding terms for full normalization.
#                        if normalization == 'full':
#                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2.0 * oe.contract("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2.0 * oe.contract("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])
#
#                    if orbitals == 'non-canonical':
#                        # Computing the Hartree-Fock term of the AAT.
#                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Computing first terms of the AATs.
#                        AAT_1[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])
#
#                        # Computing the second term of the AATs.
#                        AAT_2[lambda_alpha][beta] += 0
#
#                        # Computing the third term of the AATs.
#                        AAT_3[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,kjab,ki", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
#                        AAT_3[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijcb,ac", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)
#
#                        # Computing the fourth term of the AATs.
#                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
#                        #I = oe.contract("kjab,km->mjab", 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o])
#                        #I = oe.contract("mjab,im->ijab", I, U_R[o_, o] + half_S_core[a][o, o_].T)
#                        #AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab", t2, I)
#
#                        AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#                        #I = oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#                        #AAT_4[lambda_alpha][beta] += N**2 * 2.0 * oe.contract("ijab,ijab", t2, 2*t2 - t2.swapaxes(2,3)) * I
#                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
#                        #I = oe.contract("imab,em->ieab", 2*t2 - t2.swapaxes(2,3), U_R[v_, o_] + half_S_core[a][o_, v_].T)
#                        #I = oe.contract("ieab,ej->ijab", I, U_H[beta][v_, o_])
#                        #AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijab", t2, I)
#                        AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#                        #I = oe.contract("ijae,em->ijam", 2*t2 - t2.swapaxes(2,3), U_R[v_, o] + half_S_core[a][o, v_].T)
#                        #I = oe.contract("ijam,bm->ijab", I, U_H[beta][v_, o])
#                        #AAT_4[lambda_alpha][beta] -= N**2 * 2.0 * oe.contract("ijab,ijab", t2, I)
#
#                        # Adding terms for full normalization.
#                        if normalization == 'full':
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1.0 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])
#
#        print("Hartree-Fock AAT:")
#        print(AAT_HF, "\n")
#        print("AAT Term 1:")
#        print(AAT_1, "\n")
#        print("AAT Term 2:")
#        print(AAT_2, "\n")
#        print("AAT Term 3:")
#        print(AAT_3, "\n")
#        print("AAT Term 4:")
#        print(AAT_4, "\n")
#
#        AAT = AAT_HF + AAT_1 + AAT_2 + AAT_3 + AAT_4 + AAT_Norm
#
#        return AAT
#
#
#
#    def compute_CISD_AATs(self, normalization='full', orbitals='non-canonical', print_level=0):
#        # Compute T2 amplitudes and MP2 energy.
#        wfn_CISD = ci_wfn(self.parameters, self.wfn)
#        E_CISD, t1, t2 = wfn_CISD.solve_CISD()
#
#        # Setting initial variables for readability.
#        C = self.C
#        nbf = self.wfn.nbf
#        no = self.wfn.ndocc
#        nv = self.wfn.nbf - self.wfn.ndocc
#
#        # Setting up slices.
#        C_list, I_list = get_slices(self.parameters, self.wfn)
#        f_ = C_list[0]
#        o_ = C_list[1]
#        v_ = C_list[2]
#        t_ = C_list[3]
#
#        o = slice(0, no)
#        v = slice(no, nbf)
#        t = slice(0, nbf)
#
#        # Create a Psi4 matrix object for obtaining the perturbed MO basis integrals.
#        C_p4 = psi4.core.Matrix.from_array(C)
#
#        # Set the atom lists for Hessian.
#        natom = self.H.molecule.natom()
#        atoms = np.arange(0, natom)
#
#        # Compute the core Hamiltonian in the MO basis.
#        h = oe.contract('mp,mn,nq->pq', np.conjugate(C), self.H.T + self.H.V, C)
#
#        # Compute the electron repulsion integrals in the MO basis.
#        ERI = oe.contract('mnlg,gs->mnls', self.H.ERI, C)
#        ERI = oe.contract('mnls,lr->mnrs', ERI, np.conjugate(C))
#        ERI = oe.contract('nq,mnrs->mqrs', C, ERI)
#        ERI = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI)
#
#        # Swap axes for Dirac notation.
#        ERI = ERI.swapaxes(1,2)                 # (pr|qs) -> <pq|rs>
#
#        # Compute the Fock matrix in the MO basis.
#        F = h + oe.contract('piqi->pq', 2 * ERI[:,o,:,o] - ERI.swapaxes(2,3)[:,o,:,o])
#
#        # Use the MintsHelper to get the AO integrals from Psi4.
#        mints = psi4.core.MintsHelper(self.H.basis_set)
#        Nuc_Gradient = self.H.molecule.nuclear_repulsion_energy_deriv1().np
#
#        # Set up the atomic axial tensor.
#        AAT = np.zeros((natom * 3, 3))
#
#        # Setting up different components of the AATs.
#        AAT_HF = np.zeros((natom * 3, 3))
#        AAT_S0 = np.zeros((natom * 3, 3))
#        AAT_0S = np.zeros((natom * 3, 3))
#        AAT_SS = np.zeros((natom * 3, 3))
#        AAT_DS = np.zeros((natom * 3, 3))
#        AAT_SD = np.zeros((natom * 3, 3))
#        AAT_DD = np.zeros((natom * 3, 3))
#        AAT_Norm = np.zeros((natom * 3, 3))
#
#        # Compute normalization factor.
#        if normalization == 'intermediate':
#            N = 1
#        elif normalization == 'full':
#            N = 1 / np.sqrt(1 + 2*oe.contract('ia,ia', t1, t1) + oe.contract('ijab,ijab', t2, 2*t2 - t2.swapaxes(2,3)))
#
#        # Set up derivative t-amplitude matrices.
#        dT1_dH = []
#        dT2_dH = []
#
#        # Set up U-coefficient matrices for AAT calculations.
#        U_H = []
#
#        # Compute OPD and TPD matrices for use in computing the energy gradient.
#        # Compute normalize amplitudes.
#        N = 1 / np.sqrt(1**2 + 2*oe.contract('ia,ia->', np.conjugate(t1), t1) + oe.contract('ijab,ijab->', np.conjugate(t2), 2*t2-t2.swapaxes(2,3)))
#        t0_n = N.copy()
#        t1_n = t1 * N
#        t2_n = t2 * N
#
#        # Build OPD.
#        D_pq = np.zeros_like(F)
#        D_pq[o_,o_] -= 2 * oe.contract('ja,ia->ij', np.conjugate(t1_n), t1_n) + 2 * oe.contract('jkab,ikab->ij', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
#        D_pq[v_,v_] += 2 * oe.contract('ia,ib->ab', np.conjugate(t1_n), t1_n) + 2 * oe.contract('ijac,ijbc->ab', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t2_n)
#        D_pq[o_,v_] += 2 * np.conjugate(t0_n) * t1_n + 2 * oe.contract('jb,ijab->ia', np.conjugate(t1_n), t2_n - t2_n.swapaxes(2,3))
#        D_pq[v_,o_] += 2 * np.conjugate(t1_n.T) * t0_n + 2 * oe.contract('ijab,jb->ai', np.conjugate(t2_n - t2_n.swapaxes(2,3)), t1_n)
#        D_pq = D_pq[t_,t_]
#
#        # Build TPD.
#        D_pqrs = np.zeros_like(ERI)
#        D_pqrs[o_,o_,o_,o_] += oe.contract('klab,ijab->ijkl', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
#        D_pqrs[v_,v_,v_,v_] += oe.contract('ijab,ijcd->abcd', np.conjugate(t2_n), (2*t2_n - t2_n.swapaxes(2,3)))
#        D_pqrs[o_,v_,v_,o_] += 4 * oe.contract('ja,ib->iabj', np.conjugate(t1_n), t1_n)
#        D_pqrs[o_,v_,o_,v_] -= 2 * oe.contract('ja,ib->iajb', np.conjugate(t1_n), t1_n)
#        D_pqrs[v_,o_,o_,v_] += 2 * oe.contract('jkac,ikbc->aijb', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), 2*t2_n - t2_n.swapaxes(2,3))
#
#        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkac,ikbc->aibj', np.conjugate(t2_n), t2_n)
#        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkac,ikcb->aibj', np.conjugate(t2_n), t2_n)
#        D_pqrs[v_,o_,v_,o_] += 2 * oe.contract('jkca,ikbc->aibj', np.conjugate(t2_n), t2_n)
#        D_pqrs[v_,o_,v_,o_] -= 4 * oe.contract('jkca,ikcb->aibj', np.conjugate(t2_n), t2_n)
#
#        D_pqrs[o_,o_,v_,v_] += np.conjugate(t0_n) * (2*t2_n -t2_n.swapaxes(2,3))
#        D_pqrs[v_,v_,o_,o_] += np.conjugate(2*t2_n.swapaxes(0,2).swapaxes(1,3) - t2_n.swapaxes(2,3).swapaxes(0,2).swapaxes(1,3)) * t0_n
#        D_pqrs[v_,o_,v_,v_] += 2 * oe.contract('ja,ijcb->aibc', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
#        D_pqrs[o_,v_,o_,o_] -= 2 * oe.contract('kjab,ib->iajk', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
#        D_pqrs[v_,v_,v_,o_] += 2 * oe.contract('jiab,jc->abci', np.conjugate(2*t2_n - t2_n.swapaxes(2,3)), t1_n)
#        D_pqrs[o_,o_,o_,v_] -= 2 * oe.contract('kb,ijba->ijka', np.conjugate(t1_n), 2*t2_n - t2_n.swapaxes(2,3))
#        D_pqrs = D_pqrs[t_,t_,t_,t_]
#
#        # Compute the perturbation-independent A matrix for the CPHF coefficients with complex wavefunctions.
#        A_mag = -(2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
#        A_mag = A_mag.swapaxes(1,2)
#        G_mag = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A_mag[v,o,v,o]
#        G_mag = np.linalg.inv(G_mag.reshape((nv*no,nv*no)))
#
#        # Get the magnetic dipole AO integrals and transform into the MO basis.
#        mu_mag_AO = mints.ao_angular_momentum()
#        for a in range(3):
#            mu_mag_AO[a] = -0.5 * mu_mag_AO[a].np
#            mu_mag = oe.contract('mp,mn,nq->pq', np.conjugate(C), mu_mag_AO[a], C)
#
#            # Computing skeleton (core) first derivative integrals.
#            h_core = mu_mag
#
#            # Compute the perturbation-dependent B matrix for the CPHF coefficients with respect to a magnetic field.
#            B = h_core[v,o]
#
#            # Solve for the independent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
#            U_h = np.zeros((nbf,nbf))
#            U_h[v,o] += (G_mag @ B.reshape((nv*no))).reshape(nv,no)
#            U_h[o,v] += U_h[v,o].T
#
#            # Solve for the dependent-pairs of the CPHF U-coefficient matrix with respect to a magnetic field.
#            if self.parameters['freeze_core'] == True or orbitals == 'canonical':
#                D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
#                B = - h_core[o,o].copy() + oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
#                U_h[o,o] += B/D
#
#                D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
#                B = - h_core[v,v].copy() + oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
#                U_h[v,v] += B/D
#
#                for j in range(no):
#                    U_h[j,j] = 0
#                for c in range(no,nbf):
#                    U_h[c,c] = 0
#
#            if orbitals == 'non-canonical':
#                U_h[f_,f_] = 0
#                U_h[o_,o_] = 0
#                U_h[v_,v_] = 0
#
#            # Computing the gradient of the Fock matrix with respect to a magnetic field.
#            df_dH = np.zeros((nbf,nbf))
#
#            df_dH[o,o] -= h_core[o,o].copy()
#            df_dH[o,o] += U_h[o,o] * self.wfn.eps[o].reshape(-1,1) - U_h[o,o].swapaxes(0,1) * self.wfn.eps[o]
#            df_dH[o,o] += oe.contract('em,iejm->ij', U_h[v,o], A_mag.swapaxes(1,2)[o,v,o,o])
#
#            df_dH[v,v] -= h_core[v,v].copy()
#            df_dH[v,v] += U_h[v,v] * self.wfn.eps[v].reshape(-1,1) - U_h[v,v].swapaxes(0,1) * self.wfn.eps[v]
#            df_dH[v,v] += oe.contract('em,aebm->ab', U_h[v,o], A_mag.swapaxes(1,2)[v,v,v,o])
#
#            # Computing the gradient of the ERIs with respect to a magnetic field. # Swapaxes on these elements
#            dERI_dH =  oe.contract('tr,pqts->pqrs', U_h[:,t], ERI[t,t,:,t])
#            dERI_dH += oe.contract('ts,pqrt->pqrs', U_h[:,t], ERI[t,t,t,:])
#            dERI_dH -= oe.contract('tp,tqrs->pqrs', U_h[:,t], ERI[:,t,t,t])
#            dERI_dH -= oe.contract('tq,ptrs->pqrs', U_h[:,t], ERI[t,:,t,t])
#
#            # Compute CISD energy gradient.
#            dE_dH = oe.contract('pq,pq->', df_dH[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dH[t_,t_,t_,t_], D_pqrs)
#
#            # Computing the HF energy gradient.
#            dE_dH_HF = 2 * oe.contract('ii->', h_core[o,o])
#            dE_dH_tot = dE_dH + dE_dH_HF
#
#            # Compute dT1_dR guess amplitudes.
#            dt1_dH = -dE_dH * t1 
#            dt1_dH -= oe.contract('ji,ja->ia', df_dH[o_,o_], t1)
#            dt1_dH += oe.contract('ab,ib->ia', df_dH[v_,v_], t1)
#            dt1_dH += oe.contract('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
#            dt1_dH += oe.contract('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
#            dt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
#            dt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)
#            dt1_dH /= wfn_CISD.D_ia
#
#            # Compute dT2_dR guess amplitudes.
#            dt2_dH = -dE_dH * t2 
#            dt2_dH += oe.contract('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
#            dt2_dH += oe.contract('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
#            dt2_dH -= oe.contract('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
#            dt2_dH -= oe.contract('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
#            dt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
#            dt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
#            dt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
#            dt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
#            dt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
#            dt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
#            dt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
#            dt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
#            dt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
#            dt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
#            dt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
#            dt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)
#            dt2_dH /= wfn_CISD.D_ijab
#
#            # Solve for initial CISD energy gradient.
#            dE_dH_proj =  2.0 * oe.contract('ia,ia->', t1, df_dH[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
#            dE_dH_proj += 2.0 * oe.contract('ia,ia->', dt1_dH, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
#            dt1_dH = dt1_dH.copy()
#            dt2_dH = dt2_dH.copy()
#
#            # Setting up DIIS arrays for the error matrices and Fock matrices.
#            if self.parameters['DIIS']:
#                dt_dH_iter = []
#                de_dH_iter = []
#
#            # Start iterative procedure.
#            iteration = 1
#            while iteration <= self.parameters['max_iterations']:
#                dE_dH_proj_old = dE_dH_proj
#                dt1_dH_old = dt1_dH.copy()
#                dt2_dH_old = dt2_dH.copy()
#
#                # Solving for the derivative residuals.
#                dRt1_dH = df_dH.copy().swapaxes(0,1)[o_,v_]
#
#                dRt1_dH -= dE_dH_proj * t1
#                dRt1_dH -= oe.contract('ji,ja->ia', df_dH[o_,o_], t1)
#                dRt1_dH += oe.contract('ab,ib->ia', df_dH[v_,v_], t1)
#                dRt1_dH += oe.contract('jabi,jb->ia', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t1)
#                dRt1_dH += oe.contract('jb,ijab->ia', df_dH[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
#                dRt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dH[v_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[v_,o_,v_,v_], t2)
#                dRt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dH[o_,o_,o_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,o_,v_], t2)
#
#                dRt1_dH -= E_CISD * dt1_dH
#                dRt1_dH -= oe.contract('ji,ja->ia', F[o_,o_], dt1_dH)
#                dRt1_dH += oe.contract('ab,ib->ia', F[v_,v_], dt1_dH)
#                dRt1_dH += oe.contract('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dH)
#                dRt1_dH += oe.contract('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dH - dt2_dH.swapaxes(2,3))
#                dRt1_dH += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dH)
#                dRt1_dH -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dH)
#
#                dRt2_dH = dERI_dH.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
#
#                dRt2_dH -= dE_dH_proj * t2
#                dRt2_dH += oe.contract('abcj,ic->ijab', dERI_dH[v_,v_,v_,o_], t1)
#                dRt2_dH += oe.contract('abic,jc->ijab', dERI_dH[v_,v_,o_,v_], t1)
#                dRt2_dH -= oe.contract('kbij,ka->ijab', dERI_dH[o_,v_,o_,o_], t1)
#                dRt2_dH -= oe.contract('akij,kb->ijab', dERI_dH[v_,o_,o_,o_], t1)
#                dRt2_dH += oe.contract('ac,ijcb->ijab', df_dH[v_,v_], t2)
#                dRt2_dH += oe.contract('bc,ijac->ijab', df_dH[v_,v_], t2)
#                dRt2_dH -= oe.contract('ki,kjab->ijab', df_dH[o_,o_], t2)
#                dRt2_dH -= oe.contract('kj,ikab->ijab', df_dH[o_,o_], t2)
#                dRt2_dH += oe.contract('klij,klab->ijab', dERI_dH[o_,o_,o_,o_], t2)
#                dRt2_dH += oe.contract('abcd,ijcd->ijab', dERI_dH[v_,v_,v_,v_], t2)
#                dRt2_dH -= oe.contract('kbcj,ikca->ijab', dERI_dH[o_,v_,v_,o_], t2)
#                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
#                dRt2_dH -= oe.contract('kbic,kjac->ijab', dERI_dH[o_,v_,o_,v_], t2)
#                dRt2_dH -= oe.contract('kaci,kjbc->ijab', dERI_dH[o_,v_,v_,o_], t2)
#                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dH[o_,v_,v_,o_] - dERI_dH.swapaxes(2,3)[o_,v_,v_,o_], t2)
#                dRt2_dH -= oe.contract('kajc,ikcb->ijab', dERI_dH[o_,v_,o_,v_], t2)
#
#                dRt2_dH -= E_CISD * dt2_dH
#                dRt2_dH += oe.contract('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dH)
#                dRt2_dH += oe.contract('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dH)
#                dRt2_dH -= oe.contract('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dH)
#                dRt2_dH -= oe.contract('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dH)
#                dRt2_dH += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dH)
#                dRt2_dH += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dH)
#                dRt2_dH -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dH)
#                dRt2_dH -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dH)
#                dRt2_dH += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dH)
#                dRt2_dH += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dH)
#                dRt2_dH -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dH)
#                dRt2_dH += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
#                dRt2_dH -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dH)
#                dRt2_dH -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dH)
#                dRt2_dH += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dH)
#                dRt2_dH -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dH)
#
#                dt1_dH += dRt1_dH / wfn_CISD.D_ia
#                dt2_dH += dRt2_dH / wfn_CISD.D_ijab
#
#                # Perform DIIS extrapolation.
#                if self.parameters['DIIS']:
#                    occ = len(dt1_dH)
#                    vir = len(dt1_dH[0])
#                    dt1_dH_flat = len(np.reshape(dt1_dH, (-1)))
#                    dt2_dH_flat = len(np.reshape(dt2_dH, (-1)))
#                    res_vec = np.concatenate((np.reshape(dRt1_dH, (-1)), np.reshape(dRt2_dH, (-1))))
#                    t_vec = np.concatenate((np.reshape(dt1_dH, (-1)), np.reshape(dt2_dH, (-1))))
#                    t_vec = solve_general_DIIS(self.parameters, res_vec, t_vec, de_dH_iter, dt_dH_iter)
#                    dt1_dH = np.reshape(t_vec[0:dt1_dH_flat], (occ, vir))
#                    dt2_dH = np.reshape(t_vec[dt1_dH_flat:], (occ, occ, vir, vir))
#
#                # Compute new CISD energy gradient.
#                dE_dH_proj =  2.0 * oe.contract('ia,ia->', t1, df_dH[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dH[o_,o_,v_,v_] - dERI_dH.swapaxes(2,3)[o_,o_,v_,v_])
#                dE_dH_proj += 2.0 * oe.contract('ia,ia->', dt1_dH, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dH, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
#
#                # Compute new total energy gradient.
#                dE_dH_tot_proj = dE_dH_proj + dE_dH_HF
#
#                # Compute convergence data.
#                rms_dt1_dH = oe.contract('ia,ia->', dt1_dH_old - dt1_dH, dt1_dH_old - dt1_dH)
#                rms_dt1_dH = np.sqrt(rms_dt1_dH)
#
#                rms_dt2_dH = oe.contract('ijab,ijab->', dt2_dH_old - dt2_dH, dt2_dH_old - dt2_dH)
#                rms_dt2_dH = np.sqrt(rms_dt2_dH)
#                delta_dE_dH_proj = dE_dH_proj_old - dE_dH_proj
#
#                if print_level > 0:
#                    print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dH_proj, dE_dH_tot_proj, delta_dE_dH_proj, rms_dt1_dH, rms_dt2_dH))
#
#                if iteration > 1:
#                    if abs(delta_dE_dH_proj) < self.parameters['e_convergence'] and rms_dt1_dH < self.parameters['d_convergence'] and rms_dt2_dH < self.parameters['d_convergence']:
#                        #print("Convergence criteria met.")
#                        break
#                if iteration == self.parameters['max_iterations']:
#                    if abs(delta_dE_dH_proj) > self.parameters['e_convergence'] or rms_dt1_dH > self.parameters['d_convergence'] or rms_dt2_dH > self.parameters['d_convergence']:
#                        print("Not converged.")
#                iteration += 1
#
#            dT1_dH.append(dt1_dH)
#            dT2_dH.append(dt2_dH)
#            U_H.append(U_h)
#
#        # Delete excess variables.
#        #del dERI_dH; del dt1_dH; del dt2_dH; del dRt1_dH; del dRt2_dH; del dt1_dH_old; del dt2_dH_old
#        #del df_dH; del h_core; del B; del U_h; del A_mag; del G_mag
#        #gc.collect()
#
#
#        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
#        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
#        A = A.swapaxes(1,2)
#        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
#        G = np.linalg.inv(G.reshape((nv*no,nv*no)))
#
#        # Compute and store first derivative integrals.
#        for N1 in atoms:
#            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
#            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
#            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
#            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
#
#            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
#            ERI_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
#
#            # Compute the half derivative overlap for AAT calculation.
#            half_S_core = mints.mo_overlap_half_deriv1('LEFT', N1, C_p4, C_p4)
#
#            for a in range(3):
#                # Convert the Psi4 matrices to numpy matrices.
#                T_core[a] = T_core[a].np
#                V_core[a] = V_core[a].np
#                S_core[a] = S_core[a].np
#
#                ERI_core[a] = ERI_core[a].np
#                ERI_core[a] = ERI_core[a].swapaxes(1,2)
#                half_S_core[a] = half_S_core[a].np
#
#                # Computing skeleton (core) first derivative integrals.
#                h_core = T_core[a] + V_core[a]
#                F_core = T_core[a] + V_core[a] + oe.contract('piqi->pq', 2 * ERI_core[a][:,o,:,o] - ERI_core[a].swapaxes(2,3)[:,o,:,o])
#
#                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
#                B = -F_core[v,o] + oe.contract('ai,ii->ai', S_core[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_core[a][o,o], A.swapaxes(1,2)[v,o,o,o])
#
#                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
#                U_R = np.zeros((nbf,nbf))
#                U_R[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
#                U_R[o,v] -= U_R[v,o].T + S_core[a][o,v]
#
#                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
#                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
#                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
#                    B = F_core[o,o].copy() - oe.contract('ij,jj->ij', S_core[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
#                    U_R[o,o] += B/D
#
#                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
#                    B = F_core[v,v].copy() - oe.contract('ab,bb->ab', S_core[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
#                    U_R[v,v] += B/D
#
#                    for j in range(no):
#                        U_R[j,j] = -0.5 * S_core[a][j,j]
#                    for c in range(no,nbf):
#                        U_R[c,c] = -0.5 * S_core[a][c,c]
#
#                if orbitals == 'non-canonical':
#                    U_R[f_,f_] = -0.5 * S_core[a][f_,f_]
#                    U_R[o_,o_] = -0.5 * S_core[a][o_,o_]
#                    U_R[v_,v_] = -0.5 * S_core[a][v_,v_]
#
#                # Computing the gradient of the Fock matrix.
#                df_dR = np.zeros((nbf,nbf))
#
#                df_dR[o,o] += F_core[o,o].copy()
#                df_dR[o,o] += U_R[o,o] * self.wfn.eps[o].reshape(-1,1) + U_R[o,o].swapaxes(0,1) * self.wfn.eps[o]
#                df_dR[o,o] += oe.contract('em,iejm->ij', U_R[v,o], A.swapaxes(1,2)[o,v,o,o])
#                df_dR[o,o] -= 0.5 * oe.contract('mn,imjn->ij', S_core[a][o,o], A.swapaxes(1,2)[o,o,o,o])
#
#                df_dR[v,v] += F_core[v,v].copy()
#                df_dR[v,v] += U_R[v,v] * self.wfn.eps[v].reshape(-1,1) + U_R[v,v].swapaxes(0,1) * self.wfn.eps[v]
#                df_dR[v,v] += oe.contract('em,aebm->ab', U_R[v,o], A.swapaxes(1,2)[v,v,v,o])
#                df_dR[v,v] -= 0.5 * oe.contract('mn,ambn->ab', S_core[a][o,o], A.swapaxes(1,2)[v,o,v,o])
#
#                # Computing the gradient of the ERIs.
#                dERI_dR = ERI_core[a].copy()
#                dERI_dR += oe.contract('tp,tqrs->pqrs', U_R[:,t], ERI[:,t,t,t])
#                dERI_dR += oe.contract('tq,ptrs->pqrs', U_R[:,t], ERI[t,:,t,t])
#                dERI_dR += oe.contract('tr,pqts->pqrs', U_R[:,t], ERI[t,t,:,t])
#                dERI_dR += oe.contract('ts,pqrt->pqrs', U_R[:,t], ERI[t,t,t,:])
#
#                # Compute CISD energy gradient.
#                dE_dR = oe.contract('pq,pq->', df_dR[t_,t_], D_pq) + oe.contract('pqrs,pqrs->', dERI_dR[t_,t_,t_,t_], D_pqrs)
#
#                # Computing the HF energy gradient.
#                dE_dR_HF = 2 * oe.contract('ii->', h_core[o,o])
#                dE_dR_HF += oe.contract('ijij->', 2 * ERI_core[a][o,o,o,o] - ERI_core[a].swapaxes(2,3)[o,o,o,o])
#                dE_dR_HF -= 2 * oe.contract('ii,i->', S_core[a][o,o], self.wfn.eps[o])
#                dE_dR_HF += Nuc_Gradient[N1][a]
#
#                dE_dR_tot = dE_dR + dE_dR_HF
#
#                # Compute dT1_dR guess amplitudes.
#                dt1_dR = -dE_dR * t1
#                dt1_dR -= oe.contract('ji,ja->ia', df_dR[o_,o_], t1)
#                dt1_dR += oe.contract('ab,ib->ia', df_dR[v_,v_], t1)
#                dt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1) 
#                dt1_dR += oe.contract('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
#                dt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
#                dt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)
#                dt1_dR /= wfn_CISD.D_ia
#
#                # Compute dT2_dR guess amplitudes.
#                dt2_dR = -dE_dR * t2
#                dt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1) 
#                dt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1) 
#                dt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1) 
#                dt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1) 
#                dt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2) 
#                dt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2) 
#                dt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2) 
#                dt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2) 
#                dt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2) 
#                dt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)    
#                dt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2) 
#                dt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
#                dt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
#                dt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
#                dt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2) 
#                dt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)
#                dt2_dR /= wfn_CISD.D_ijab
#
#                # Solve for initial CISD energy gradient.
#                dE_dR_proj =  2.0 * oe.contract('ia,ia->', t1, df_dR[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
#                dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
#                dt1_dR = dt1_dR.copy()
#                dt2_dR = dt2_dR.copy()                
#
#                # Setting up DIIS arrays for the error matrices and Fock matrices.
#                if self.parameters['DIIS']:
#                    dt_dR_iter = [] 
#                    de_dR_iter = [] 
#
#                # Start iterative procedure.
#                iteration = 1
#                while iteration <= self.parameters['max_iterations']:
#                    dE_dR_proj_old = dE_dR_proj
#                    dt1_dR_old = dt1_dR.copy()
#                    dt2_dR_old = dt2_dR.copy()
#
#                    # Solving for the derivative residuals.
#                    dRt1_dR = df_dR.copy().swapaxes(0,1)[o_,v_]
#
#                    dRt1_dR -= dE_dR_proj * t1
#                    dRt1_dR -= oe.contract('ji,ja->ia', df_dR[o_,o_], t1)
#                    dRt1_dR += oe.contract('ab,ib->ia', df_dR[v_,v_], t1)
#                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t1)
#                    dRt1_dR += oe.contract('jb,ijab->ia', df_dR[o_,v_], 2.0 * t2 - t2.swapaxes(2,3))
#                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * dERI_dR[v_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[v_,o_,v_,v_], t2)
#                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * dERI_dR[o_,o_,o_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,o_,v_], t2)
#
#                    dRt1_dR -= E_CISD * dt1_dR
#                    dRt1_dR -= oe.contract('ji,ja->ia', F[o_,o_], dt1_dR)
#                    dRt1_dR += oe.contract('ab,ib->ia', F[v_,v_], dt1_dR)
#                    dRt1_dR += oe.contract('jabi,jb->ia', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt1_dR)
#                    dRt1_dR += oe.contract('jb,ijab->ia', F[o_,v_], 2.0 * dt2_dR - dt2_dR.swapaxes(2,3))
#                    dRt1_dR += oe.contract('ajbc,ijbc->ia', 2.0 * ERI[v_,o_,v_,v_] - ERI.swapaxes(2,3)[v_,o_,v_,v_], dt2_dR)
#                    dRt1_dR -= oe.contract('kjib,kjab->ia', 2.0 * ERI[o_,o_,o_,v_] - ERI.swapaxes(2,3)[o_,o_,o_,v_], dt2_dR)
#
#                    dRt2_dR = dERI_dR.copy().swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]
#
#                    dRt2_dR -= dE_dR_proj * t2
#                    dRt2_dR += oe.contract('abcj,ic->ijab', dERI_dR[v_,v_,v_,o_], t1)
#                    dRt2_dR += oe.contract('abic,jc->ijab', dERI_dR[v_,v_,o_,v_], t1)
#                    dRt2_dR -= oe.contract('kbij,ka->ijab', dERI_dR[o_,v_,o_,o_], t1)
#                    dRt2_dR -= oe.contract('akij,kb->ijab', dERI_dR[v_,o_,o_,o_], t1)
#                    dRt2_dR += oe.contract('ac,ijcb->ijab', df_dR[v_,v_], t2)
#                    dRt2_dR += oe.contract('bc,ijac->ijab', df_dR[v_,v_], t2)
#                    dRt2_dR -= oe.contract('ki,kjab->ijab', df_dR[o_,o_], t2)
#                    dRt2_dR -= oe.contract('kj,ikab->ijab', df_dR[o_,o_], t2)
#                    dRt2_dR += oe.contract('klij,klab->ijab', dERI_dR[o_,o_,o_,o_], t2)
#                    dRt2_dR += oe.contract('abcd,ijcd->ijab', dERI_dR[v_,v_,v_,v_], t2)
#                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', dERI_dR[o_,v_,v_,o_], t2)
#                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
#                    dRt2_dR -= oe.contract('kbic,kjac->ijab', dERI_dR[o_,v_,o_,v_], t2)
#                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', dERI_dR[o_,v_,v_,o_], t2)
#                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * dERI_dR[o_,v_,v_,o_] - dERI_dR.swapaxes(2,3)[o_,v_,v_,o_], t2)
#                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', dERI_dR[o_,v_,o_,v_], t2)
#
#                    dRt2_dR -= E_CISD * dt2_dR
#                    dRt2_dR += oe.contract('abcj,ic->ijab', ERI[v_,v_,v_,o_], dt1_dR)
#                    dRt2_dR += oe.contract('abic,jc->ijab', ERI[v_,v_,o_,v_], dt1_dR)
#                    dRt2_dR -= oe.contract('kbij,ka->ijab', ERI[o_,v_,o_,o_], dt1_dR)
#                    dRt2_dR -= oe.contract('akij,kb->ijab', ERI[v_,o_,o_,o_], dt1_dR)
#                    dRt2_dR += oe.contract('ac,ijcb->ijab', F[v_,v_], dt2_dR)
#                    dRt2_dR += oe.contract('bc,ijac->ijab', F[v_,v_], dt2_dR)
#                    dRt2_dR -= oe.contract('ki,kjab->ijab', F[o_,o_], dt2_dR)
#                    dRt2_dR -= oe.contract('kj,ikab->ijab', F[o_,o_], dt2_dR)
#                    dRt2_dR += oe.contract('klij,klab->ijab', ERI[o_,o_,o_,o_], dt2_dR)
#                    dRt2_dR += oe.contract('abcd,ijcd->ijab', ERI[v_,v_,v_,v_], dt2_dR)
#                    dRt2_dR -= oe.contract('kbcj,ikca->ijab', ERI[o_,v_,v_,o_], dt2_dR)
#                    dRt2_dR += oe.contract('kaci,kjcb->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
#                    dRt2_dR -= oe.contract('kbic,kjac->ijab', ERI[o_,v_,o_,v_], dt2_dR)
#                    dRt2_dR -= oe.contract('kaci,kjbc->ijab', ERI[o_,v_,v_,o_], dt2_dR)
#                    dRt2_dR += oe.contract('kbcj,ikac->ijab', 2.0 * ERI[o_,v_,v_,o_] - ERI.swapaxes(2,3)[o_,v_,v_,o_], dt2_dR)
#                    dRt2_dR -= oe.contract('kajc,ikcb->ijab', ERI[o_,v_,o_,v_], dt2_dR)
#
#                    dt1_dR += dRt1_dR / wfn_CISD.D_ia
#                    dt2_dR += dRt2_dR / wfn_CISD.D_ijab
#
#                    # Perform DIIS extrapolation.
#                    if self.parameters['DIIS']:
#                        occ = len(dt1_dR)
#                        vir = len(dt1_dR[0])
#                        dt1_dR_flat = len(np.reshape(dt1_dR, (-1)))
#                        dt2_dR_flat = len(np.reshape(dt2_dR, (-1)))
#                        res_vec = np.concatenate((np.reshape(dRt1_dR, (-1)), np.reshape(dRt2_dR, (-1))))
#                        t_vec = np.concatenate((np.reshape(dt1_dR, (-1)), np.reshape(dt2_dR, (-1))))
#                        t_vec = solve_general_DIIS(self.parameters, res_vec, t_vec, de_dR_iter, dt_dR_iter)
#                        dt1_dR = np.reshape(t_vec[0:dt1_dR_flat], (occ, vir))
#                        dt2_dR = np.reshape(t_vec[dt1_dR_flat:], (occ, occ, vir, vir))
#
#                    # Compute new CISD energy gradient.
#                    dE_dR_proj =  2.0 * oe.contract('ia,ia->', t1, df_dR[o_,v_]) + oe.contract('ijab,ijab->', t2, 2.0 * dERI_dR[o_,o_,v_,v_] - dERI_dR.swapaxes(2,3)[o_,o_,v_,v_])
#                    dE_dR_proj += 2.0 * oe.contract('ia,ia->', dt1_dR, F[o_,v_]) + oe.contract('ijab,ijab->', dt2_dR, 2.0 * ERI[o_,o_,v_,v_] - ERI.swapaxes(2,3)[o_,o_,v_,v_])
#
#                    # Compute new total energy gradient.
#                    dE_dR_tot_proj = dE_dR_proj + dE_dR_HF
#
#                    # Compute convergence data.
#                    rms_dt1_dR = oe.contract('ia,ia->', dt1_dR_old - dt1_dR, dt1_dR_old - dt1_dR) 
#                    rms_dt1_dR = np.sqrt(rms_dt1_dR)
#
#                    rms_dt2_dR = oe.contract('ijab,ijab->', dt2_dR_old - dt2_dR, dt2_dR_old - dt2_dR) 
#                    rms_dt2_dR = np.sqrt(rms_dt2_dR)
#                    delta_dE_dR_proj = dE_dR_proj_old - dE_dR_proj
#
#                    if print_level > 0:
#                        print(" %02d %20.12f %20.12f %20.12f %20.12f %20.12f" % (iteration, dE_dR_proj, dE_dR_tot_proj, delta_dE_dR_proj, rms_dt1_dR, rms_dt2_dR))
#
#                    if iteration > 1:
#                        if abs(delta_dE_dR_proj) < self.parameters['e_convergence'] and rms_dt1_dR < self.parameters['d_convergence'] and rms_dt2_dR < self.parameters['d_convergence']:
#                            #print("Convergence criteria met.")
#                            break
#                    if iteration == self.parameters['max_iterations']:
#                        if abs(delta_dE_dR_proj) > self.parameters['e_convergence'] or rms_dt1_dR > self.parameters['d_convergence'] or rms_dt2_dR > self.parameters['d_convergence']:
#                            print("Not converged.")
#                    iteration += 1
#
#                # Compute derivative of the normalization factor.
#                N_R = - (1 / np.sqrt((1 + 2*oe.contract('ia,ia', np.conjugate(t1), t1) + oe.contract('ijab,ijab', np.conjugate(t2), 2*t2 - t2.swapaxes(2,3)))**3))
#                N_R *= 0.5 * (2*oe.contract('ia,ia', np.conjugate(dt1_dR), t1) + 2*oe.contract('ia,ia', dt1_dR, np.conjugate(t1)) + oe.contract('ijab,ijab', np.conjugate(dt2_dR), 2*t2 - t2.swapaxes(2,3)) + oe.contract('ijab,ijab', dt2_dR, np.conjugate(2*t2 - t2.swapaxes(2,3))))
#
#                for beta in range(0,3):
#                    #Setting up AAT indexing.
#                    lambda_alpha = 3 * N1 + a
#
#                    if orbitals == 'canonical':
#                        # Computing the Hartree-Fock term of the AAT.
#                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Singles/Refence terms.
#                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_H[beta][v_,o_])
#
#                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T)
#                        AAT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core[a][o,o_].T)
#
#                        # Reference/Singles terms.
#                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ck", dT1_dH[beta], U_R[v_,o_] + half_S_core[a][o_,v_].T)
#
#                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fk", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core[a][o_,v_].T) # Canonical
#                        AAT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,cn", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core[a][o,v_].T)                
#
#                        # Singles/Singles terms.
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dH[beta])
#
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,cf,kf", dt1_dR, U_H[beta][v_,v_], t1) # Canonical
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,nk,nc", dt1_dR, U_H[beta][o_,o_], t1) # Canonical
#
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], U_R[o_,o_] + half_S_core[a][o_,o_].T, t1)
#
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,fa,ka", t1, U_H[beta][v_,v_], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1) # Canonical
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fc,ik,if", t1, U_H[beta][v_,v_], U_R[o_,o_] + half_S_core[a][o_,o_].T, t1) # Canonical
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ca,na", t1, U_H[beta][o_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1) # Canonical
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,kn,in,ic", t1, U_H[beta][o_,o], U_R[o_,o] + half_S_core[a][o,o_].T, t1)
#                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,kc,ia,ia", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core[a][v_,o_].T, t1)
#                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,fn,kc", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core[a][o,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,fk,nc", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,cn,kf", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core[a][o,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("kc,fn,ck,nf", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
#
#                        # Doubles/Singles terms.
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)
#
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,ia,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3))
#
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fc,ia,ikaf", t1, U_H[beta][v_,v_], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical
#                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,kn,ia,inac", t1, U_H[beta][o_,o_], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical
#                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,ik,incf", t1, U_H[beta][v_,o_], U_R[o_,o_] + half_S_core[a][o_,o_].T, 2*t2 - t2.swapaxes(2,3))
#                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("kc,fn,in,ikfc", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core[a][o,o_].T, 2*t2 - t2.swapaxes(2,3))
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,ca,knaf", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("kc,fn,fa,knca", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
#
#                        # Singles/Doubles terms.
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))
#
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,dl,kc", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
#
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ea,kice", t1, U_H[beta][o_,v_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
#                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,kc,im,kmca", t1, U_H[beta][o_,v_], U_R[o_,o_] + half_S_core[a][o_,o_].T, 2*t2 - t2.swapaxes(2,3))
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ac,em,imce", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ec,em,imac", t1, U_H[beta][v_,v_], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical
#                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,ki,em,kmae", t1, U_H[beta][o_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, 2*t2 - t2.swapaxes(2,3)) # Canonical
#                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core[a][o,v_].T, 2*t2 - t2.swapaxes(2,3))
#
#                        # Doubles/Doubles terms.
#                        AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])
#
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dt2_dR - dt2_dR.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical
#
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("klcd,mlcd,mk", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("klcd,kled,ce", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)
#
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ec,ea", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, v_], U_R[v_, v_] + half_S_core[a][v_, v_].T) # Canonical
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Adding terms for full normalization. 
#                        if normalization == 'full':
#                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("ijab,kjab,ki", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][o_, o_]) # Canonical
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,ijcb,ac", 2*t2 - t2.swapaxes(2,3), t2, U_H[beta][v_, v_]) # Canonical
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])
#
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ai", t1, U_H[beta][v_, o_])
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("kc,kc", t1, U_H[beta][o_, v_])
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("kc,cf,kf", t1, U_H[beta][v_,v_], t1) # Canonical
#                            AAT_Norm[lambda_alpha][beta] -= N * N_R * 2 * oe.contract("kc,nk,nc", t1, U_H[beta][o_,o_], t1) # Canonical
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,kc,ikac", t1, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))
#
#                    if orbitals == 'non-canonical':
#                        # Computing the Hartree-Fock term of the AAT.
#                        AAT_HF[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Singles/Refence terms.
#                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_H[beta][v_,o_])
#
#                        AAT_S0[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T)
#                        AAT_S0[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core[a][o,o_].T)
#
#                        # Reference/Singles terms.
#                        AAT_0S[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dT1_dH[beta], U_R[v_,o_] + half_S_core[a][o_,v_].T)
#
#                        AAT_0S[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,im,am", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core[a][o,v_].T)
#
#                        # Singles/Singles terms.
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dH[beta])
#
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], U_R[v_,v_] + half_S_core[a][v_,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], U_R[o_,o_] + half_S_core[a][o_,o_].T, t1)
#
#                        AAT_SS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,im,km,ka", t1, U_H[beta][o_,o], U_R[o_,o] + half_S_core[a][o,o_].T, t1)
#                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,ia,kc,kc", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core[a][v_,o_].T, t1)
#                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,em,ia", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core[a][o,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,ei,ma", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,am,ie", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core[a][o,v_].T, t1)
#                        AAT_SS[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,ai,me", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
#
#                        # Doubles/Singles terms.
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)
#
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core[a][v_,o_].T, 2*t2 - t2.swapaxes(2,3))
#
#                        AAT_DS[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,km,kiea", t1, U_H[beta][v_,o], U_R[o_,o] + half_S_core[a][o,o_].T, 2*t2 - t2.swapaxes(2,3))
#                        AAT_DS[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,em,ec,imac", t1, U_H[beta][v_,o_], U_R[v_,v_] + half_S_core[a][v_,v_].T, 2*t2 - t2.swapaxes(2,3))
#
#                        # Singles/Doubles terms.
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))
#
#                        AAT_SD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core[a][o_,v_].T, t1)
#
#                        AAT_SD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,km,em,kiea", t1, U_H[beta][o_,o], U_R[v_,o] + half_S_core[a][o,v_].T, 2*t2 - t2.swapaxes(2,3))
#
#                        # Doubles/Doubles terms.
#                        AAT_DD[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])
#
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[o_, o_] + half_S_core[a][o_, o_].T)
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, U_R[v_, v_] + half_S_core[a][v_, v_].T)
#
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,kjab,km,im", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][o_, o], U_R[o_, o] + half_S_core[a][o, o_].T)
#                        AAT_DD[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core[a][o_, v_].T)
#                        AAT_DD[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core[a][o, v_].T)
#
#                        # Adding terms for full normalization. 
#                        if normalization == 'full':
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])
#
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ia,ai", t1, U_H[beta][v_, o_])
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
#                            AAT_Norm[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)
#
#        print("Hartree-Fock AAT:")
#        print(AAT_HF, "\n")
#        print("Singles/Reference AAT:")
#        print(AAT_S0, "\n")
#        print("Reference/Singles AAT:")
#        print(AAT_0S, "\n")
#        print("Singles/Singles AAT:")
#        print(AAT_SS, "\n")
#        print("Doubles/Singles:")
#        print(AAT_DS, "\n")
#        print("Singles/Doubles:")
#        print(AAT_SD, "\n")
#        print("Doubles/Doubles:")
#        print(AAT_DD, "\n")
#
#        AAT = AAT_HF + AAT_S0 + AAT_0S + AAT_SS + AAT_DS + AAT_SD + AAT_DD + AAT_Norm
#
#        return AAT
















