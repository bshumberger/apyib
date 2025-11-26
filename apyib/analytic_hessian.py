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



    def compute_RHF_Hessian(self, orbitals='non-canonical'):
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
        N = self.H.molecule.nuclear_repulsion_energy_deriv2().np

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Set up the storage matrices for Hessian calculations.
        U_R = []
        h_R = []
        ERI_R = []
        S_R = []
        F_R = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        # Compute and store first derivative integrals.
        for N1 in atoms:
            # Compute the skeleton (core) one-electron first derivative integrals in the MO basis.
            T_a = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_a = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_a = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)

            # Compute the skeleton (core) two-electron first derivative integrals in the MO basis.
            ERI_a = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)

            for a in range(3):
                # Convert the Psi4 matrices to numpy matrices.
                T_a[a] = T_a[a].np
                V_a[a] = V_a[a].np
                S_a[a] = S_a[a].np

                ERI_a[a] = ERI_a[a].np
                ERI_a[a] = ERI_a[a].swapaxes(1,2)

                # Computing skeleton (core) first derivative integrals.
                h_a = T_a[a] + V_a[a]
                F_a = T_a[a] + V_a[a] + oe.contract('piqi->pq', 2 * ERI_a[a][:,o,:,o] - ERI_a[a].swapaxes(2,3)[:,o,:,o])

                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                B = -F_a[v,o] + oe.contract('ai,ii->ai', S_a[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_a[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                U_a = np.zeros((nbf,nbf))
                U_a[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                U_a[o,v] -= U_a[v,o].T + S_a[a][o,v]

                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                    B = F_a[o,o].copy() - oe.contract('ij,jj->ij', S_a[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_a[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_a[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                    U_a[o,o] += B/D

                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                    B = F_a[v,v].copy() - oe.contract('ab,bb->ab', S_a[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_a[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_a[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                    U_a[v,v] += B/D

                    for j in range(no):
                        U_a[j,j] = -0.5 * S_a[a][j,j]
                    for b in range(no,nbf):
                        U_a[b,b] = -0.5 * S_a[a][b,b]

                if orbitals == 'non-canonical':
                    U_a[f_,f_] = -0.5 * S_a[a][f_,f_]
                    U_a[o_,o_] = -0.5 * S_a[a][o_,o_]
                    U_a[v_,v_] = -0.5 * S_a[a][v_,v_]

                # Appending to lists.
                h_R.append(h_a)
                ERI_R.append(ERI_a[a])
                S_R.append(S_a[a])
                F_R.append(F_a)
                U_R.append(U_a)

        for N1 in atoms:
            for N2 in atoms:
                # Compute the nuclear skeleton (core) one-electron second derivative integrals in the MO basis.
                T_ab = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_ab = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_ab = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)

                # Compute the nuclear skeleton (core) two-electron second derivative integrals in the MO basis.
                ERI_ab = mints.mo_tei_deriv2(N1, N2, C_p4, C_p4, C_p4, C_p4)

                for a in range(3):
                    for b in range(3):
                        ab = 3 * a + b
                        N1a = 3 * N1 + a
                        N2b = 3 * N2 + b
                        # Convert the Psi4 matrices to numpy matrices.
                        T_RR = T_ab[ab].np
                        V_RR = V_ab[ab].np
                        S_RR = S_ab[ab].np

                        h_RR = T_RR + V_RR

                        ERI_RR = ERI_ab[ab].np
                        ERI_RR = ERI_RR.swapaxes(1,2)

                        #eta_RR = oe.contract('im,jm->ij', U_R[N1a][o,:], U_R[N2b][o,:]) + oe.contract('im,jm->ij', U_R[N2b][o,:], U_R[N1a][o,:]) - oe.contract('im,jm->ij', S_R[N1a][o,:], S_R[N2b][o,:]) - oe.contract('im,jm->ij', S_R[N2b][o,:], S_R[N1a][o,:])

                        # Asymmetric approach to Hessian calculation.
                        Hessian[N1a][N2b] += 2 * oe.contract('ii->', h_RR[o,o])
                        Hessian[N1a][N2b] += 1 * oe.contract('ijij->', 2 * ERI_RR[o,o,o,o] - ERI_RR[o,o,o,o].swapaxes(2,3))
                        Hessian[N1a][N2b] += 2 * oe.contract('pi,pi->', U_R[N2b][:,o], F_R[N1a][:,o] + F_R[N1a][o,:].T) 
                        Hessian[N1a][N2b] -= 2 * oe.contract('pi,pj,ij->', U_R[N2b][:,o], S_R[N1a][:,o], F[o,o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('pj,ip,ij->', U_R[N2b][:,o], S_R[N1a][o,:], F[o,o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,ij->', S_RR[o,o], F[o,o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,ij->', S_R[N1a][o,o], F_R[N2b][o,o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,ki,kj->', S_R[N1a][o,o], U_R[N2b][o,o], F[o,o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,kj,ik->', S_R[N1a][o,o], U_R[N2b][o,o], F[o,o])
                        Hessian[N1a][N2b] -= 2 * oe.contract('ij,pk,ipjk->', S_R[N1a][o,o],  U_R[N2b][:,o], 2 * ERI[o,:,o,o] + 2 * ERI[o,o,o,:].swapaxes(1,3) - ERI[o,:,o,o].swapaxes(2,3) - ERI[o,o,:,o].swapaxes(1,2).swapaxes(2,3))

                        # Symmetric approach to Hessian calculation.
                        #Hessian[N1a][N2b] += 2 * oe.contract('ii->', h_RR[o,o])
                        #Hessian[N1a][N2b] += 1 * oe.contract('ijij->', 2 * ERI_RR[o,o,o,o] - ERI_RR[o,o,o,o].swapaxes(2,3))
                        #Hessian[N1a][N2b] -= 2 * oe.contract('ii,i->', S_RR[o,o], self.wfn.eps[o]) 
                        #Hessian[N1a][N2b] -= 2 * oe.contract('ii,i->', eta_RR[o,o], self.wfn.eps[o])
                        #Hessian[N1a][N2b] += 4 * oe.contract('ij,ij->', U_R[N1a][:,o], F_R[N2b][:,o]) + 4 * oe.contract('ij,ij->', U_R[N2b][:,o], F_R[N1a][:,o])
                        #Hessian[N1a][N2b] += 4 * oe.contract('ij,ij,i->', U_R[N1a][:,o], U_R[N2b][:,o], self.wfn.eps[:])
                        #Hessian[N1a][N2b] += 4 * oe.contract('ij,kl,ikjl->', U_R[N1a][:,o], U_R[N2b][:,o], 4 * ERI[:,:,o,o] - ERI[:,:,o,o].swapaxes(2,3) - ERI[:,o,:,o].swapaxes(1,2).swapaxes(2,3))

        # Add second derivative of the nuclear repulsion energy.
        Hessian += N

        return Hessian



    def compute_RHF_Hessian_opt(self, orbitals='non-canonical'):
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

        # Use the MintsHelper to get the AO integrals from Psi4.
        mints = psi4.core.MintsHelper(self.H.basis_set)
        N = self.H.molecule.nuclear_repulsion_energy_deriv2().np

        # Set up the Hessian.
        Hessian = np.zeros((natom * 3, natom * 3))

        # Set up a list of CPHF matrices for each perturbation to keep track of perturbations and avoid duplicate computations.
        U_R = []
        U_R_list = []

        # Compute the perturbation-independent A matrix for the CPHF coefficients with real wavefunctions.
        A = (2 * ERI - ERI.swapaxes(2,3)) + (2 * ERI - ERI.swapaxes(2,3)).swapaxes(1,3)
        A = A.swapaxes(1,2)
        G = oe.contract('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v,v].reshape(nv,1,nv,1) - F[o,o].reshape(1,no,1,no)) + A[v,o,v,o]
        G = np.linalg.inv(G.reshape((nv*no,nv*no)))

        for N1 in atoms:
            # Compute the skeleton (core) first derivative integrals in the MO basis.
            T_Ra = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_Ra = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_Ra = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            ERI_Ra = mints.ao_tei_deriv1(N1)

            for N2 in range(N1, natom):
                # Compute the skeleton (core) first derivative integrals in the MO basis.
                T_Rb = mints.mo_oei_deriv1('KINETIC', N2, C_p4, C_p4)
                V_Rb = mints.mo_oei_deriv1('POTENTIAL', N2, C_p4, C_p4)
                S_Rb = mints.mo_oei_deriv1('OVERLAP', N2, C_p4, C_p4)
                ERI_Rb = mints.ao_tei_deriv1(N2)

                # Compute the nuclear skeleton (core) second derivative integrals in the MO basis.
                T_RaRb = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_RaRb = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_RaRb = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)
                ERI_RaRb = mints.ao_tei_deriv2(N1, N2)

                for a in range(3):
                    N1a = 3 * N1 + a
                    # Convert the Psi4 matrices to numpy matrices.
                    if type(T_Ra[a]) == psi4.core.Matrix:
                        T_Ra[a] = T_Ra[a].np
                        V_Ra[a] = V_Ra[a].np
                        S_Ra[a] = S_Ra[a].np
                        ERI_Ra[a] = ERI_Ra[a].np

                        # Compute the skeleton (core) two-electron MO integrals.
                        ERI_Ra[a] = oe.contract('mnlg,gs->mnls', ERI_Ra[a], C)
                        ERI_Ra[a] = oe.contract('mnls,lr->mnrs', ERI_Ra[a], np.conjugate(C))
                        ERI_Ra[a] = oe.contract('nq,mnrs->mqrs', C, ERI_Ra[a])
                        ERI_Ra[a] = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI_Ra[a])
                        ERI_Ra[a] = ERI_Ra[a].swapaxes(1,2)

                    # Computing skeleton (core) first derivative integrals.
                    h_Ra = T_Ra[a] + V_Ra[a]
                    F_Ra = T_Ra[a] + V_Ra[a] + oe.contract('piqi->pq', 2 * ERI_Ra[a][:,o,:,o] - ERI_Ra[a].swapaxes(2,3)[:,o,:,o])

                    if N1a in U_R_list:
                        ind = U_R_list.index(N1a)
                        U_Ra = U_R[ind]

                    else:
                        # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                        B = -F_Ra[v,o] + oe.contract('ai,ii->ai', S_Ra[a][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_Ra[a][o,o], A.swapaxes(1,2)[v,o,o,o])

                        # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                        U_Ra = np.zeros((nbf,nbf))
                        U_Ra[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                        U_Ra[o,v] -= U_Ra[v,o].T + S_Ra[a][o,v]

                        # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                        if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                            D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                            B = F_Ra[o,o].copy() - oe.contract('ij,jj->ij', S_Ra[a][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_Ra[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_Ra[a][o,o], A.swapaxes(1,2)[o,o,o,o])
                            U_Ra[o,o] += B/D

                            D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                            B = F_Ra[v,v].copy() - oe.contract('ab,bb->ab', S_Ra[a][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_Ra[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_Ra[a][o,o], A.swapaxes(1,2)[v,o,v,o])
                            U_Ra[v,v] += B/D

                            for j in range(no):
                                U_Ra[j,j] = -0.5 * S_Ra[a][j,j]
                            for c in range(no,nbf):
                                U_Ra[c,c] = -0.5 * S_Ra[a][c,c]

                        if orbitals == 'non-canonical':
                            U_Ra[f_,f_] = -0.5 * S_Ra[a][f_,f_]
                            U_Ra[o_,o_] = -0.5 * S_Ra[a][o_,o_]
                            U_Ra[v_,v_] = -0.5 * S_Ra[a][v_,v_]

                        # Appending CPHF matrices to list.
                        U_R.append(U_Ra)
                        U_R_list.append(N1a)

                    for b in range(3):
                        ab = 3 * a + b 
                        N2b = 3 * N2 + b
                        
                        if N2b < N1a:
                            Hessian[N1a][N2b] = 0

                        elif N2b >= N1a:
                            # Convert the first derivative Psi4 matrices to numpy matrices.
                            if type(T_Rb[b]) == psi4.core.Matrix:
                                T_Rb[b] = T_Rb[b].np
                                V_Rb[b] = V_Rb[b].np
                                S_Rb[b] = S_Rb[b].np
                                ERI_Rb[b] = ERI_Rb[b].np
        
                                # Compute the skeleton (core) two-electron MO integrals.
                                ERI_Rb[b] = oe.contract('mnlg,gs->mnls', ERI_Rb[b], C)
                                ERI_Rb[b] = oe.contract('mnls,lr->mnrs', ERI_Rb[b], np.conjugate(C))
                                ERI_Rb[b] = oe.contract('nq,mnrs->mqrs', C, ERI_Rb[b])
                                ERI_Rb[b] = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI_Rb[b])
                                ERI_Rb[b] = ERI_Rb[b].swapaxes(1,2)

                            # Computing skeleton (core) first derivative integrals.
                            h_Rb = T_Rb[b] + V_Rb[b]
                            F_Rb = T_Rb[b] + V_Rb[b] + oe.contract('piqi->pq', 2 * ERI_Rb[b][:,o,:,o] - ERI_Rb[b].swapaxes(2,3)[:,o,:,o])

                            # Convert the second derivative Psi4 matrices to numpy matrices.
                            if type(T_RaRb[ab]) == psi4.core.Matrix:
                                T_RaRb[ab] = T_RaRb[ab].np
                                V_RaRb[ab] = V_RaRb[ab].np
                                S_RaRb[ab] = S_RaRb[ab].np
                                ERI_RaRb[ab] = ERI_RaRb[ab].np

                                # Compute the skeleton (core) two-electron MO integrals.
                                ERI_RaRb[ab] = oe.contract('mnlg,gs->mnls', ERI_RaRb[ab], C)
                                ERI_RaRb[ab] = oe.contract('mnls,lr->mnrs', ERI_RaRb[ab], np.conjugate(C))
                                ERI_RaRb[ab] = oe.contract('nq,mnrs->mqrs', C, ERI_RaRb[ab])
                                ERI_RaRb[ab] = oe.contract('mp,mqrs->pqrs', np.conjugate(C), ERI_RaRb[ab])
                                ERI_RaRb[ab] = ERI_RaRb[ab].swapaxes(1,2)

                            # Computing skeleton (core) second derivative integral.
                            h_RaRb = T_RaRb[ab] + V_RaRb[ab]

                            if N2b in U_R_list:
                                ind = U_R_list.index(N2b)
                                U_Rb = U_R[ind]

                            else:
                                # Compute the perturbation-dependent B matrix for the CPHF coefficients.
                                B = -F_Rb[v,o] + oe.contract('ai,ii->ai', S_Rb[b][v,o], F[o,o]) + 0.5 * oe.contract('mn,amin->ai', S_Rb[b][o,o], A.swapaxes(1,2)[v,o,o,o])

                                # Solve for the independent-pairs of the CPHF U-coefficient matrix.
                                U_Rb = np.zeros((nbf,nbf))
                                U_Rb[v,o] += (G @ B.reshape((nv*no))).reshape(nv,no)
                                U_Rb[o,v] -= U_Rb[v,o].T + S_Rb[b][o,v]

                                # Solve for the dependent-pairs of the CPHF U-coefficient matrix.
                                if self.parameters['freeze_core'] == True or orbitals == 'canonical':
                                    D = (self.wfn.eps[o] - self.wfn.eps[o].reshape(-1,1)) + np.eye(no)
                                    B = F_Rb[o,o].copy() - oe.contract('ij,jj->ij', S_Rb[b][o,o], F[o,o]) + oe.contract('em,iejm->ij', U_Rb[v,o], A.swapaxes(1,2)[o,v,o,o]) - 0.5 * oe.contract('mn,imjn->ij', S_Rb[b][o,o], A.swapaxes(1,2)[o,o,o,o])
                                    U_Rb[o,o] += B/D

                                    D = (self.wfn.eps[v] - self.wfn.eps[v].reshape(-1,1)) + np.eye(nv)
                                    B = F_Rb[v,v].copy() - oe.contract('ab,bb->ab', S_Rb[b][v,v], F[v,v]) + oe.contract('em,aebm->ab', U_Rb[v,o], A.swapaxes(1,2)[v,v,v,o]) - 0.5 * oe.contract('mn,ambn->ab', S_Rb[b][o,o], A.swapaxes(1,2)[v,o,v,o])
                                    U_Rb[v,v] += B/D

                                    for j in range(no):
                                        U_Rb[j,j] = -0.5 * S_Rb[b][j,j]
                                    for c in range(no,nbf):
                                        U_Rb[c,c] = -0.5 * S_Rb[b][c,c]

                                if orbitals == 'non-canonical':
                                    U_Rb[f_,f_] = -0.5 * S_Rb[b][f_,f_]
                                    U_Rb[o_,o_] = -0.5 * S_Rb[b][o_,o_]
                                    U_Rb[v_,v_] = -0.5 * S_Rb[b][v_,v_]

                                # Appending CPHF matrices to list.
                                U_R.append(U_Rb)
                                U_R_list.append(N2b)

                            #print("U_R_list:", U_R_list)

                            eta_RR = oe.contract('im,jm->ij', U_Ra[o,:], U_Rb[o,:]) + oe.contract('im,jm->ij', U_Rb[o,:], U_Ra[o,:]) - oe.contract('im,jm->ij', S_Ra[a][o,:], S_Rb[b][o,:]) - oe.contract('im,jm->ij', S_Rb[b][o,:], S_Ra[a][o,:])

                            # Asymmetric approach to Hessian calculation.
                            #Hessian[N1a][N2b] += 2 * oe.contract('ii->', h_RaRb[o,o])
                            #Hessian[N1a][N2b] += 1 * oe.contract('ijij->', 2 * ERI_RaRb[ab][o,o,o,o] - ERI_RaRb[ab][o,o,o,o].swapaxes(2,3))
                            #Hessian[N1a][N2b] += 2 * oe.contract('pi,pi->', U_Rb[:,o], F_Ra[:,o] + F_Ra[o,:].T) 
                            #Hessian[N1a][N2b] -= 2 * oe.contract('pi,pj,ij->', U_Rb[:,o], S_Ra[a][:,o], F[o,o])
                            #Hessian[N1a][N2b] -= 2 * oe.contract('pj,ip,ij->', U_Rb[:,o], S_Ra[a][o,:], F[o,o])
                            #Hessian[N1a][N2b] -= 2 * oe.contract('ij,ij->', S_RaRb[ab][o,o], F[o,o])
                            #Hessian[N1a][N2b] -= 2 * oe.contract('ij,ij->', S_Ra[a][o,o], F_Rb[o,o])
                            #Hessian[N1a][N2b] -= 2 * oe.contract('ij,ki,kj->', S_Ra[a][o,o], U_Rb[o,o], F[o,o])
                            #Hessian[N1a][N2b] -= 2 * oe.contract('ij,kj,ik->', S_Ra[a][o,o], U_Rb[o,o], F[o,o])
                            #Hessian[N1a][N2b] -= 2 * oe.contract('ij,pk,ipjk->', S_Ra[a][o,o],  U_Rb[:,o], 2 * ERI[o,:,o,o] + 2 * ERI[o,o,o,:].swapaxes(1,3) - ERI[o,:,o,o].swapaxes(2,3) - ERI[o,o,:,o].swapaxes(1,2).swapaxes(2,3))

                            # Symmetric approach to Hessian calculation.
                            Hessian[N1a][N2b] += 2 * oe.contract('ii->', h_RaRb[o,o])
                            Hessian[N1a][N2b] += 1 * oe.contract('ijij->', 2 * ERI_RaRb[ab][o,o,o,o] - ERI_RaRb[ab][o,o,o,o].swapaxes(2,3))
                            Hessian[N1a][N2b] -= 2 * oe.contract('ii,i->', S_RaRb[ab][o,o], self.wfn.eps[o]) 
                            Hessian[N1a][N2b] -= 2 * oe.contract('ii,i->', eta_RR[o,o], self.wfn.eps[o])
                            Hessian[N1a][N2b] += 4 * oe.contract('ij,ij->', U_Ra[:,o], F_Rb[:,o]) + 4 * oe.contract('ij,ij->', U_Rb[:,o], F_Ra[:,o])
                            Hessian[N1a][N2b] += 4 * oe.contract('ij,ij,i->', U_Ra[:,o], U_Rb[:,o], self.wfn.eps[:])
                            Hessian[N1a][N2b] += 4 * oe.contract('ij,kl,ikjl->', U_Ra[:,o], U_Rb[:,o], 4 * ERI[:,:,o,o] - ERI[:,:,o,o].swapaxes(2,3) - ERI[:,o,:,o].swapaxes(1,2).swapaxes(2,3))


        Hessian += Hessian.T
        Hessian -= 0.5 * np.eye(3*natom) * Hessian

        # Add second derivative of the nuclear repulsion energy.
        Hessian += N

        return Hessian
















