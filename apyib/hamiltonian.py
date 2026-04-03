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

        # ======================== Electric and Magnetic Fields ========================
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

        # ======================== Phase-Space Hamiltonian ========================
        # Add phase space componets to core Hamiltonian for a phase space Hamiltonian.
        PS = parameters.get('hamiltonian', None)
        if PS == 'phase-space':
            natom = self.molecule.natom()
            nbf = self.basis_set.nbf()

            # Building Gamma' (G1).
            # Compute (delta_BA + delta_CA) component.
            d = np.zeros((natom,nbf,nbf))
            for A in range(natom):
                for mu in range(nbf):
                    for nu in range(nbf):
                        if self.basis_set.function_to_center(mu) == A:
                            d[A][mu][nu] += 1
                        if self.basis_set.function_to_center(nu) == A:
                            d[A][mu][nu] += 1

            # Compute linear momentum integrals.
            self.nabla = mints.ao_nabla()   # p_{\mu\nu} = \int \phi_{\mu}^*(r) \frac{\partial}{\partial r} \phi_{\nu}(r) dr
            self.nabla = np.array(self.nabla)

            # Compute the electron translation factor (ETF) Gamma'.
            self.G1 = -0.5 * oe.contract('bmn,amn->abmn', self.nabla, d)

            # TEST: Test the sum of G1 is equal to the negative of the electronic momentum, eq.19.
            assert np.allclose(oe.contract('abmn->bmn', self.G1) + self.nabla, 0, atol=1e-12)

            # Building Gamma'' (G2).
            _me = psi4.qcel.constants.get("electron mass") # kg
            _na = psi4.qcel.constants.get("Avogadro constant") # 1/mol
            _u = 1/(1000 * _na) # kg

            X, M, elem, Z, uniq = self.molecule.to_arrays()
            X = np.array(X)
            M = np.array(M / (_me / _u)) # Converting to electron mass.
            self.M = M
            w = 0.3

            # Compute the locality factor, zeta.
            zeta = np.zeros((natom,nbf,nbf))
            for A in range(natom):
                for mu in range(nbf):
                    B = self.basis_set.function_to_center(mu)
                    for nu in range(nbf):
                        C = self.basis_set.function_to_center(nu)
                        if A == B and A == C:
                            zeta[A][mu][nu] += 1 # Chosen due to behavior of the function as it goes to zero, i.e. e^0 = 1.
                        else:
                            zeta[A][mu][nu] += np.exp(-w * (2 * np.linalg.norm(X[A] - X[B])**2 * np.linalg.norm(X[A] - X[C])**2)/(np.linalg.norm(X[A] - X[B])**2 + np.linalg.norm(X[A] - X[C])**2))

            # Compute the recentering component, X_0.
            X_0 = np.zeros((3,nbf,nbf))
            sum_zeta = oe.contract('amn->mn', zeta)
            X_0 = oe.contract('amn,ab->bmn', zeta, X) / sum_zeta

            # TEST: Test the recentering condition, eq. 44.
            assert np.allclose(oe.contract('amn,ab->bmn', zeta, X) - oe.contract('amn,bmn->bmn', zeta, X_0), 0, atol=1e-12)

            # Compute recentered position, X_rcntr.
            X_rcntr = np.zeros((natom,3,nbf,nbf))
            for A in range(natom):
                for alpha in range(3):
                    X_rcntr[A][alpha] += X[A][alpha] - X_0[alpha]

            # Compute the locality weighted massless momenta of intertia, K.
            K = np.zeros((3,3,nbf,nbf))
            K_int1 = - oe.contract('amn,abmn,abmn->mn', zeta, X_rcntr, X_rcntr)
            K_int2 = oe.contract('amn,abmn,acmn->bcmn', zeta, X_rcntr, X_rcntr)
            for alpha in range(3):
                K[alpha][alpha] += K_int1
            K += K_int2

            # Compute K inverse with indices ordered: mu, nu, alpha, beta.
            K_inv = np.swapaxes(np.swapaxes(K, 0, 2), 1, 3)
            for mu in range(nbf):
                for nu in range(nbf):
                    #print(K_inv[mu][nu])
                    K_inv[mu][nu] = np.linalg.inv(K_inv[mu][nu])
                    #print(K_inv[mu][nu])

            # Compute the electronic momentum around atom A, l.
            l = np.zeros((natom,3,nbf,nbf))
            self.L = mints.ao_angular_momentum()
            self.L = np.array(self.L)
            for A in range(natom):
                l[A][0] += self.L[0] - (X[A][1] * self.nabla[2] - X[A][2] * self.nabla[1])
                l[A][1] += self.L[1] - (X[A][2] * self.nabla[0] - X[A][0] * self.nabla[2])
                l[A][2] += self.L[2] - (X[A][0] * self.nabla[1] - X[A][1] * self.nabla[0])

            # Compute the atomic orbital centered angular momentum, J.
            J = np.zeros((3,nbf,nbf))
            for alpha in range(3):
                for mu in range(nbf):
                    B = self.basis_set.function_to_center(mu)
                    for nu in range(nbf):
                        C = self.basis_set.function_to_center(nu)
                        J[alpha][mu][nu] -= 0.5 * (l[B][alpha][mu][nu] + l[C][alpha][mu][nu])

            # Compute the electron rotation factor (ERF) Gamma''.
            K_inv_J = np.zeros((nbf,nbf, 3))
            J_swap = np.swapaxes(np.swapaxes(J, 0, 1), 1, 2)
            for mu in range(nbf):
                for nu in range(nbf):
                    K_inv_J[mu][nu] += K_inv[mu][nu] @ J_swap[mu][nu]
            K_inv_J = np.swapaxes(np.swapaxes(K_inv_J, 1, 2), 0, 1)

            self.G2 = np.zeros((natom,3,nbf,nbf))
            for A in range(natom):
                for mu in range(nbf):
                    for nu in range(nbf):
                        self.G2[A][0][mu][nu] += zeta[A][mu][nu] * (X_rcntr[A][1][mu][nu] * K_inv_J[2][mu][nu] - X_rcntr[A][2][mu][nu] * K_inv_J[1][mu][nu])
                        self.G2[A][1][mu][nu] += zeta[A][mu][nu] * (X_rcntr[A][2][mu][nu] * K_inv_J[0][mu][nu] - X_rcntr[A][0][mu][nu] * K_inv_J[2][mu][nu])
                        self.G2[A][2][mu][nu] += zeta[A][mu][nu] * (X_rcntr[A][0][mu][nu] * K_inv_J[1][mu][nu] - X_rcntr[A][1][mu][nu] * K_inv_J[0][mu][nu])

            # TEST: Test the conservation of angular momentum, eq. 48.
            mom_con = oe.contract('abmn->bmn', self.G2)
            assert np.allclose(mom_con, 0, atol=1e-12)

            # TEST: Test relationship between Gamma'' and J, eq. 51.
            TEST = np.zeros((3,nbf,nbf))
            for A in range(natom):
                for mu in range(nbf):
                    for nu in range(nbf):
                        TEST[0][mu][nu] += X[A][1] * self.G2[A][2][mu][nu] - X[A][2] * self.G2[A][1][mu][nu]
                        TEST[1][mu][nu] += X[A][2] * self.G2[A][0][mu][nu] - X[A][0] * self.G2[A][2][mu][nu]
                        TEST[2][mu][nu] += X[A][0] * self.G2[A][1][mu][nu] - X[A][1] * self.G2[A][0][mu][nu]
    
            assert np.allclose(TEST - J, 0, atol=1e-12)

            # Compute total Gamma.
            self.G = self.G1 + self.G2

            # Building Gamma tilde.
            self.G_tilde = np.zeros((natom,3,nbf,nbf))

            # Compute the diagonalized overlap matrix, s.
            s, U = np.linalg.eigh(self.S)

            # Compute Gamma tilde.
            denom = np.zeros((nbf,nbf))
            for lambd in range(nbf):
                for sigma in range(nbf):
                    denom[lambd][sigma] += (s[lambd] + s[sigma])
            self.G_tilde = 2.0 * oe.contract('ms,absl,ln->abmn', U, oe.contract('lm,abmn,ns->abls', U.T, self.G, U) / denom, U.T)

            # TEST: Testing Gamma tilde's relationship to Gamma with eq. 26.
            for A in range(natom):
                for alpha in range(3):
                    GSSG = 0.5 * (self.G_tilde[A][alpha] @ self.S + self.S @ self.G_tilde[A][alpha])
                    assert np.allclose(GSSG - self.G[A][alpha], 0, atol=1e-12)

            # Compute the second derivative coupling (SDC) integral, zeta tilde.
            self.zeta_tilde = - oe.contract('abml,ls,absn->abmn', self.G_tilde, self.S, self.G_tilde)

            # Adding zeta tilde to the potential operator.
            self.V = self.V + oe.contract('abmn,a->mn', self.zeta_tilde, np.reciprocal(2.0 * M))

            # Including nuclear momentum dependent terms in the Hamiltonian for finite-difference calculations.
            if parameters.get('P_nuc', None) != None:
                P_nuc = np.array(parameters['P_nuc'])
                # Adding first derivative coupling term.
                self.V = self.V - 1.0j * oe.contract('ab,abmn,a->mn', P_nuc, self.G, np.reciprocal(M))

                # Computing nuclear kinetic energy.
                self.T_nuc = oe.contract('ab,a->', P_nuc**2, np.reciprocal(2.0 * M))


            # The implemented equations for the phase-space Hamiltonian came from two
            # manuscripts: DOI: 10.1021/acs.jctc.4c00662 and DOI: 10.1063/5.0192083









