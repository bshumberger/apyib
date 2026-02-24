"""Contains the Hamiltonian object."""

import psi4
import numpy as np
import opt_einsum as oe

class Hamiltonian(object):
    """
    Atomic orbital Hamiltonian object.
    """
    # Define the specific properties of the Hamiltonian which is dependent on the molecule.
    def __init__(self, parameters):

        # Clear previous options incase of serial calculations.
        psi4.core.clean_options()

        # Set the basis set for the calculation.
        psi4.set_options({'basis': parameters['basis']})

        # Apply frozen core approximation.
        psi4.set_options({'freeze_core': parameters['freeze_core']})

        # Define the molecule and basis set as properties of the Hamiltonian.
        self.molecule = psi4.geometry(parameters['geom'])
        self.basis_set = psi4.core.BasisSet.build(self.molecule)

        # Use the MintsHelper to get the AO integrals.
        mints = psi4.core.MintsHelper(self.basis_set)

        # Compute AO integrals.
        self.T = mints.ao_kinetic().np       # T_{\mu\nu} = \int \phi_{\mu}^*(r) \left( -\frac{1}{2} \nabla^2_r \right) \phi_{\nu}(r) dr
        self.V = mints.ao_potential().np     # V_{\mu\nu} = \int \phi_{\mu}^*(r) \left( -\sum_A^N \frac{Z}{r_A} \right) \phi_{\nu}(r) dr
        self.ERI = mints.ao_eri().np         # (\mu \nu|\lambda \sigma) = \int \phi_{\mu}^*(r_1) \phi_{\nu}(r_1) r_{12}^{-1} \phi_{\lambda}^*(r_2) \phi_{\sigma}(r_2) dr_1 dr_2
        self.S = mints.ao_overlap().np       # S_{\mu\nu} = \int \phi_{\mu}^*(r) \phi_{\nu}(r) dr

        # Checking for fields.
        E_field = False
        M_field = False
        for alpha in range(3):
            if parameters['F_el'][alpha] != 0.0:
                E_field = True
            if parameters['F_mag'][alpha] != 0.0:
                M_field = True

        # Electric dipole AO integrals.
        if E_field == True:
            self.mu_el = mints.ao_dipole()       # \mu^{el}_{\mu\nu, \alpha} = -e \int \phi_{\mu}^*(r) r_{\alpha} \phi_{\nu}(r) dr
            for alpha in range(3):
                self.mu_el[alpha] = self.mu_el[alpha].np

        # Magnetic dipole AO integrals.
        if M_field == True:
            self.mu_mag = mints.ao_angular_momentum()    # \mu^{mag}_{\mu\nu, \alpha} = - \frac(e}{2 m_e} \int \phi_{\mu}^*(r) (r x p)_{\alpha} \phi_{\nu}(r) dr
            for alpha in range(3):
                self.mu_mag[alpha] = -0.5j * self.mu_mag[alpha].np

        # Compute the nuclear repulsion energy.
        F_el = [0.0, 0.0, 0.0]
        for alpha in range(3):
            F_el[alpha] += parameters['F_el'][alpha] * -1
        self.E_nuc = self.molecule.nuclear_repulsion_energy(F_el)

        # Add electric and magnetic potentials to the core Hamiltonian.
        for alpha in range(3):
            if E_field == True:
                self.V =  self.V - parameters['F_el'][alpha] * self.mu_el[alpha]
            if M_field == True:
                self.V =  self.V - parameters['F_mag'][alpha] * self.mu_mag[alpha]

        # Add phase space componets to core Hamiltonian for a phase space Hamiltonian.
        PS = parameters.get('hamiltonian', None)
        if PS == 'Phase-Space':

            # Build Gamma (G) by constructing Gamma' (G1) and Gamma'' (G2).
            natom = self.molecule.natom()
            nbf = self.basis_set.nbf()
            self.G1 = [list(range(3)) for i in range(natom)]
            self.G2 = [list(range(3)) for i in range(natom)]

            # Building Gamma' (G1).
            # Compute (delta_BA + delta_CA) component.
            d = list(range(natom))
            for A in range(natom):
                d[A] = np.zeros((nbf, nbf))
                for mu in range(nbf):
                    for nu in range(nbf):
                        if self.basis_set.function_to_center(mu) == A:
                            d[A][mu][nu] += 1
                        if self.basis_set.function_to_center(nu) == A:
                            d[A][mu][nu] += 1

            # Compute linear momentum integrals.
            self.nabla = mints.ao_nabla()   # p_{\mu\nu} = \int \phi_{\mu}^*(r) \frac{\partial}{\partial r} \phi_{\nu}(r) dr
            for alpha in range(3):
                self.nabla[alpha] = self.nabla[alpha].np

            # Compute the electron translation factor (ETF) Gamma'.
            for A in range(natom):
                for alpha in range(3):
                    self.G1[A][alpha] = -0.5 * self.nabla[alpha] * d[A]
                    #print(A, alpha)
                    #print(self.G1[A][alpha])

            # Test the sum of G1 is equal to the electronic momentum, eq.19.
            for alpha in range(3):
                p = np.zeros((nbf,nbf))
                for A in range(natom):
                    p += self.G1[A][alpha]
                for mu in range(nbf):
                    for nu in range(nbf):
                        assert p[mu][nu] == -self.nabla[alpha][mu][nu]

            # Building Gamma'' (G2).
            X, M, elem, Z, uniq = self.molecule.to_arrays()
            self.Mass = M

            w = 0.3
            z = list(range(natom))
            for A in range(natom):
                z[A] = np.zeros((nbf, nbf))
                for mu in range(nbf):
                    B = self.basis_set.function_to_center(mu)
                    for nu in range(nbf):
                        C = self.basis_set.function_to_center(nu)

                        # Compute zeta (z).
                        if A == B and A == C:
                            z[A][mu][nu] += 1 # Chosen due to behavior of the function as it goes to zero, i.e. e^0 = 1.
                        else:
                            z[A][mu][nu] += np.exp(-w * (2 * np.linalg.norm(X[A] - X[B])**2 * np.linalg.norm(X[A] - X[C])**2)/(np.linalg.norm(X[A] - X[B])**2 + np.linalg.norm(X[A] - X[C])**2))

            # Compute the recentering component, X_0.
            X_0 = list(range(3))
            for alpha in range(3):
                X_0[alpha] = np.zeros((nbf, nbf))
                for mu in range(nbf):
                    for nu in range(nbf):
                        z_tot = 0
                        for A in range(natom):
                            X_0[alpha][mu][nu] += z[A][mu][nu] * X[A][alpha]
                            z_tot += z[A][mu][nu]
                        X_0[alpha][mu][nu] /= z_tot
                #print(X_0[alpha])

            # Test the recentering condition, eq. 44.
            cent_cond = list(range(3))
            for alpha in range(3):
                cent_cond[alpha] = np.zeros((nbf, nbf))
                for mu in range(nbf):
                    for nu in range(nbf):
                        for A in range(natom):
                            cent_cond[alpha][mu][nu] += z[A][mu][nu] * X[A][alpha] - z[A][mu][nu] * X_0[alpha][mu][nu]
                assert cent_cond[alpha].all() == 0

            # Compute recentered position (dX).
            dX = [list(range(3)) for i in range(natom)]
            for A in range(natom):
                for alpha in range(3):
                    dX[A][alpha] = X[A][alpha] - X_0[alpha]
                    #print(A, alpha, X[A][alpha])
                    #print(X_0[alpha])
                    #print(dX[A][alpha])

            # Compute the locality weighted massless momenta of intertia, K.
            K = [list(range(3)) for i in range(3)]
            for alpha in range(3):
                for beta in range(3):
                    K[alpha][beta] = np.zeros((nbf, nbf))
            for A in range(natom):
                for alpha in range(3):
                    for beta in range(3):
                        if alpha == beta:
                            for gamma in range(3):
                                K[alpha][beta] += -z[A] * dX[A][gamma] * dX[A][gamma]
                        K[alpha][beta] += z[A] * dX[A][alpha] *  dX[A][beta]
            #for alpha in range(3):
            #    for beta in range(3):
            #        print(alpha, beta)
            #        print(K[alpha][beta])

            # Compute K^-1 with indices ordered: mu, nu, alpha, beta.
            K1 = [list(range(nbf)) for i in range(nbf)]
            for mu in range(nbf):
                for nu in range(nbf):
                    K1[mu][nu] = np.zeros((3,3))
                    for alpha in range(3):
                        for beta in range(3):
                            K1[mu][nu][alpha][beta] += K[alpha][beta][mu][nu]
                    K1[mu][nu] = np.linalg.inv(K1[mu][nu])
                    #print(K1[mu][nu])

            # Compute the electronic momentum around atom A, l.
            l = [list(range(3)) for i in range(natom)]
            self.L = mints.ao_angular_momentum()
            for alpha in range(3):
                self.L[alpha] = self.L[alpha].np
            for A in range(natom):
                for alpha in range(3):
                    l[A][alpha] = np.zeros((nbf, nbf))
                l[A][0] += self.L[0] - (X[A][1] * self.nabla[2] - X[A][2] * self.nabla[1])
                l[A][1] += self.L[1] - (X[A][2] * self.nabla[0] - X[A][0] * self.nabla[2])
                l[A][2] += self.L[2] - (X[A][0] * self.nabla[1] - X[A][1] * self.nabla[0])
                #print(A)
                #print(l[A])

            # Compute the atomic orbital centered angular momentum, J.
            J = list(range(3))
            for alpha in range(3):
                J[alpha] = np.zeros((nbf, nbf))
                for mu in range(nbf):
                    B = self.basis_set.function_to_center(mu)
                    for nu in range(nbf):
                        C = self.basis_set.function_to_center(nu)
                        J[alpha][mu][nu] = -0.5 * (l[B][alpha][mu][nu] + l[C][alpha][mu][nu])
                #print(J[alpha])

            # Compute the electron rotation factor (ERF) Gamma''.
            K1J = list(range(3))
            for alpha in range(3):
                K1J[alpha] = np.zeros((nbf, nbf))
            for mu in range(nbf):
                for nu in range(nbf):
                    for alpha in range(3):
                        for beta in range(3):
                            K1J[alpha][mu][nu] += K1[mu][nu][alpha][beta] * J[beta][mu][nu]
            #print(K1J)
            for A in range(natom):
                for alpha in range(3):
                    self.G2[A][alpha] = np.zeros((nbf, nbf))
            for A in range(natom):
                for mu in range(nbf):
                    for nu in range(nbf):
                        self.G2[A][0][mu][nu] += z[A][mu][nu] * (dX[A][1][mu][nu] * K1J[2][mu][nu] - dX[A][2][mu][nu] * K1J[1][mu][nu])
                        self.G2[A][1][mu][nu] += z[A][mu][nu] * (dX[A][2][mu][nu] * K1J[0][mu][nu] - dX[A][0][mu][nu] * K1J[2][mu][nu])
                        self.G2[A][2][mu][nu] += z[A][mu][nu] * (dX[A][0][mu][nu] * K1J[1][mu][nu] - dX[A][1][mu][nu] * K1J[0][mu][nu])
                #print(self.G2[A])

            # Test conservation of angular momentum, eq. 48.
            mom_con = list(range(3))
            for alpha in range(3):
                mom_con[alpha] = np.zeros((nbf, nbf))
            for A in range(natom):
                mom_con[0] += self.G2[A][0]
                mom_con[1] += self.G2[A][1]
                mom_con[2] += self.G2[A][2]
            for alpha in range(3):
                assert mom_con[alpha].all() == 0

            # Compute total Gamma
            self.G = [list(range(3)) for i in range(natom)]
            for A in range(natom):
                for alpha in range(3):
                    self.G[A][alpha] = self.G1[A][alpha] + self.G2[A][alpha]
            #print(self.G)

            # Building Gamma tilde.
            self.G_t = [list(range(3)) for i in range(natom)]
            for A in range(natom):
                for alpha in range(3):
                    self.G_t[A][alpha] = np.zeros((nbf, nbf))

            # Compute the diagonalized overlap matrix, s.
            s, U = np.linalg.eigh(self.S)

            # Compute Gamma tilde.
            for A in range(natom):
                for alpha in range(3):
                    for mu in range(nbf):
                        for nu in range(nbf):
                            for lambd in range(nbf):
                                for sigma in range(nbf):
                                    self.G_t[A][alpha][mu][nu] += 2.0 * (U[mu][sigma] * (1/(s[lambd] + s[sigma])) * (U.T @ self.G[A][alpha] @ U)[sigma][lambd]) * U.T[lambd][nu]
                    #print(self.G_t[A][alpha])

            # Compute the second derivative coupling (SDC) term.
            self.SDC = np.zeros((nbf, nbf))
            for A in range(natom):
                for alpha in range(3):
                    self.SDC -= (self.G_t[A][alpha] @ self.S @ self.G_t[A][alpha]) / (2 * M[A]) 
            #print(SDC)

            # Adding SDC to the potential operator.
            self.V = self.V + self.SDC

            # Adding Gamma to the potential operator when P not equal to zero.
            P = parameters.get('P_nuc', None)
            if P != None:
                for A in range(natom):
                    for alpha in range(3):
                        self.V = self.V - (1.0j * P[A][alpha] * self.G[A][alpha]) / (M[A])

            # Computing nuclear kinetic energy.
            if P != None:
                self.K_nuc = 0
                for A in range(natom):
                    for alpha in range(3):
                        self.K_nuc += P[A][alpha]**2 / (2.0 * M[A])














