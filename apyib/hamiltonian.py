"""Contains the Hamiltonian object."""

import psi4
import numpy as np
import opt_einsum as oe
import time

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
        hamiltonian = parameters.get('hamiltonian', None)
        formulation = parameters.get('ps_formulation', None)
        if hamiltonian == 'phase-space':
            natom = self.molecule.natom()
            nbf = self.basis_set.nbf()

            if formulation == 'atomic-orbital':
                # Setting up the phase-space locality and partitioning parameters and partitioning prefactor.
                if parameters.get('ps_ao_locality_factor(w)', None) == None:
                    w = 0.3
                    print("Using default locality parameter, i.e. w = 0.3.")
                else:
                    w = parameters.get('ps_ao_locality_factor(w)')

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
                # Confirmed that nabla is the bare derivative integral without signs required for linear momentum integral, p. Confirmed
                # by analytic APT code which aligns with sign in CPHF notes for magnetic fields.
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
                # Confirmed angular momentum integrals do include negative sign associated with the linear momentum integral, p. Confirmed
                # by inclusion of the magnetic dipole operators in the Hamiltonian and consistent with definition of the magnetic dipole
                # operator in Stephens VCD derivation.
                l = np.zeros((natom,3,nbf,nbf))
                self.L = mints.ao_angular_momentum()
                self.L = np.array(self.L)
                for A in range(natom):
                    l[A][0] += -self.L[0] - (X[A][1] * self.nabla[2] - X[A][2] * self.nabla[1])
                    l[A][1] += -self.L[1] - (X[A][2] * self.nabla[0] - X[A][0] * self.nabla[2])
                    l[A][2] += -self.L[2] - (X[A][0] * self.nabla[1] - X[A][1] * self.nabla[0])

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

            if formulation == 'basis-free':
                #t0 = time.time()
                # Set Psi4 options for building the grid.
                psi4.set_options({'dft_spherical_points': parameters['spherical_grid_points']})
                psi4.set_options({'dft_radial_points': parameters['radial_grid_points']})
                psi4.set_options({'dft_pruning_scheme': parameters['grid_pruning_scheme']})

                basis = self.basis_set #psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvtz")
                nbf = basis.nbf()
                
                # --- Step 1: Build the grid infrastructure ---
                # Psi4 builds DFT grids through VBase, which requires a
                # density functional. We use SVWN (a simple LDA functional)
                # as a dummy — we only care about the grid, not the functional.
                sup = psi4.driver.dft.build_superfunctional("SVWN", True)[0]
                Vpot = psi4.core.VBase.build(basis, sup, "RV")
                Vpot.initialize()
                
                # --- Step 2: Build the basis function evaluator ---
                # RKSFunctions evaluates basis functions (and optionally their
                # gradients) at grid points. It needs three things:
                #   - the basis set
                #   - the max number of points in any single block
                #   - the total number of basis functions
                nblocks = Vpot.nblocks()
                npoints_max = max(Vpot.get_block(b).npoints() for b in range(nblocks))
                point_func = psi4.core.RKSFunctions(basis, npoints_max, nbf)
                
                # set_ansatz controls what Psi4 computes at each grid point:
                #   0 = LDA:  only basis function values (PHI)
                #   1 = GGA:  values + first derivatives (PHI, PHI_X, PHI_Y, PHI_Z)
                #   2 = meta: values + first and second derivatives
                point_func.set_ansatz(1)
                
                # set_deriv is redundant with set_ansatz but explicitly requests
                # the derivative level for the basis functions themselves.
                point_func.set_deriv(1)
                
                # set_pointers gives the evaluator a density matrix. We pass a
                # zero matrix because we do not need density-dependent quantities —
                # we only want the raw basis function values and gradients.
                D_zero = psi4.core.Matrix(nbf, nbf)
                point_func.set_pointers(D_zero)
                
                # --- Step 3: Loop over blocks and collect everything ---
                # Psi4 organizes grid points into "blocks" of spatially nearby
                # points. Within each block, only a subset of basis functions
                # are nonzero (those centered on nearby atoms). This is the
                # "extent screening" that makes DFT integration efficient.
                #
                # For each block, functions_local_to_global() returns the indices
                # of the basis functions that are nonzero in that block. The
                # arrays returned by basis_values() use this LOCAL indexing.
                
                all_coords = []
                all_weights = []
                all_phi = []
                all_phi_x = []
                all_phi_y = []
                all_phi_z = []

                for b in range(nblocks):
                    block = Vpot.get_block(b)
                    npts_b = block.npoints()
                
                    # Grid point coordinates (in bohr) and integration weights.
                    x = np.array(block.x())
                    y = np.array(block.y())
                    z = np.array(block.z())
                    w = np.array(block.w())
                    all_coords.append(np.column_stack([x, y, z]))
                    all_weights.append(w)
                
                    # Evaluate basis functions and gradients at this block's points.
                    point_func.compute_points(block)
                
                    # functions_local_to_global maps local index -> global basis index.
                    # Example: if lto = [0, 3, 7], then local column 0 in the PHI
                    # array corresponds to global basis function 0, local column 1
                    # corresponds to global basis function 3, etc.
                    lto = block.functions_local_to_global()
                    nlocal = len(lto)
                
                    # Extract the values. The arrays have shape (npoints_max, nlocal)
                    # but only the first npts_b rows contain data for this block.
                    phi_block   = np.array(point_func.basis_values()["PHI"])[:npts_b, :nlocal]
                    phi_x_block = np.array(point_func.basis_values()["PHI_X"])[:npts_b, :nlocal]
                    phi_y_block = np.array(point_func.basis_values()["PHI_Y"])[:npts_b, :nlocal]
                    phi_z_block = np.array(point_func.basis_values()["PHI_Z"])[:npts_b, :nlocal]
                
                    # Scatter the local arrays into full-size (npts_b, nbf) arrays.
                    # Basis functions not in lto are zero at all points in this block.
                    phi_full   = np.zeros((npts_b, nbf))
                    phi_x_full = np.zeros((npts_b, nbf))
                    phi_y_full = np.zeros((npts_b, nbf))
                    phi_z_full = np.zeros((npts_b, nbf))
                    for i_local, i_global in enumerate(lto):
                        phi_full[:, i_global]   = phi_block[:, i_local]
                        phi_x_full[:, i_global] = phi_x_block[:, i_local]
                        phi_y_full[:, i_global] = phi_y_block[:, i_local]
                        phi_z_full[:, i_global] = phi_z_block[:, i_local]
                
                    all_phi.append(phi_full)
                    all_phi_x.append(phi_x_full)
                    all_phi_y.append(phi_y_full)
                    all_phi_z.append(phi_z_full)

                # --- Step 4: Concatenate blocks into flat arrays ---
                coords  = np.vstack(all_coords)    # shape (npts_total, 3)
                weights = np.concatenate(all_weights)  # shape (npts_total,)
                phi     = np.vstack(all_phi)       # shape (npts_total, nbf)
                phi_x   = np.vstack(all_phi_x)    # shape (npts_total, nbf)
                phi_y   = np.vstack(all_phi_y)    # shape (npts_total, nbf)
                phi_z   = np.vstack(all_phi_z)    # shape (npts_total, nbf)
                
                npts_total = len(weights)
                #print(f"Grid points: {npts_total}, Basis functions: {nbf}")
                
                ## --- Step 5: Sanity check — numerical overlap vs analytic ---
                ## The overlap matrix S_mn = integral phi_m(r) phi_n(r) dr
                ## should be reproduced by numerical integration on the grid:
                ## S_mn ≈ sum_g w_g * phi_m(r_g) * phi_n(r_g)
                #mints = psi4.core.MintsHelper(basis)
                #S_analytic = mints.ao_overlap().np
                #S_numerical = np.einsum('g,gm,gn->mn', weights, phi, phi)
                #max_error = np.max(np.abs(S_analytic - S_numerical))
                #print(f"Max error in numerical overlap: {max_error:.2e}")
                #print("Analytic AO Overlap Matrix:")
                #print(S_analytic)
                #print("Numerical AO Overlap Matrix:")
                #print(S_numerical)

                # Getting constants and converting to atomic units.
                _me = psi4.qcel.constants.get("electron mass") # kg
                _na = psi4.qcel.constants.get("Avogadro constant") # 1/mol
                _u = 1/(1000 * _na) # kg

                R, M, elem, Q, uniq = self.molecule.to_arrays()
                R = np.array(R)
                M = np.array(M / (_me / _u)) # Converting to electron mass.
                self.M = M

                #print(Q/np.sum(Q))
                #print(M/np.sum(M))

                # Setting up the phase-space locality and partitioning parameters and partitioning prefactor.
                if parameters.get('ps_bf_partitioning_prefactor', None) == None:
                    prefactor = Q.copy()
                    print("Using default prefactor selection, i.e. charge.")
                elif parameters.get('ps_bf_partitioning_prefactor', None) == 'charge':
                    prefactor = Q.copy()
                elif parameters.get('ps_bf_partitioning_prefactor', None) == 'mass':
                    prefactor = M.copy()

                if parameters.get('ps_bf_locality_factor(beta)', None) == None:
                    beta = 9.0
                    print("Using default locality parameter, i.e. beta = 9.0.")
                else:
                    beta = parameters.get('ps_bf_locality_factor(beta)')

                if parameters.get('ps_bf_partitioning_factor(sigma)', None) == None:
                    sigma = 2.0
                    print("Using default partitioning parameter, i.e. sigma = 2.0.")
                else:
                    sigma = parameters.get('ps_bf_partitioning_factor(sigma)')

                # Building the space-partitioning operator, theta.
                coords_diff = coords[:, None, :] - R[None, :, :] # (npts, natom, 3)
                dist_2 = np.sum(coords_diff**2, axis=2) # (npts, natom)
                theta_numerator = prefactor[None, :] * np.exp(-dist_2 / sigma**2 ) # (npts, natom)
                theta_denominator = theta_numerator.sum(axis=1, keepdims=True)  # (npts, natom)
                theta = theta_numerator / theta_denominator # (npts, natom)

                # Numerically integrating the < mu | theta p | nu > integral.
                theta_px = oe.contract('r,rm,ra,rn->amn', weights, phi, theta, phi_x)
                theta_py = oe.contract('r,rm,ra,rn->amn', weights, phi, theta, phi_y)
                theta_pz = oe.contract('r,rm,ra,rn->amn', weights, phi, theta, phi_z)

                # Building the Gamma' integrals.
                self.G1 = np.zeros((natom,3,nbf,nbf))
                for A in range(natom):
                    self.G1[A][0] -= 0.5 * (theta_px[A] - theta_px[A].T)
                    self.G1[A][1] -= 0.5 * (theta_py[A] - theta_py[A].T)
                    self.G1[A][2] -= 0.5 * (theta_pz[A] - theta_pz[A].T)

                # Building the locality factor, zeta
                zeta = np.zeros((natom,natom))
                for A in range(natom):
                    for B in range(natom):
                        zeta[A][B] += self.M[A] * np.exp(-np.linalg.norm(R[A] - R[B])**2 / beta**2)

                # Building J.
                L = np.zeros((natom,3,nbf,nbf))
                J = np.zeros((natom,3,nbf,nbf))
                for A in range(natom):
                    L[A][0] += oe.contract('r,rm,r,r,rn->mn', weights, phi, coords_diff[:,A,1], theta[:,A], phi_z)
                    L[A][0] -= oe.contract('r,rm,r,r,rn->mn', weights, phi, coords_diff[:,A,2], theta[:,A], phi_y)
                    L[A][1] += oe.contract('r,rm,r,r,rn->mn', weights, phi, coords_diff[:,A,2], theta[:,A], phi_x)
                    L[A][1] -= oe.contract('r,rm,r,r,rn->mn', weights, phi, coords_diff[:,A,0], theta[:,A], phi_z)
                    L[A][2] += oe.contract('r,rm,r,r,rn->mn', weights, phi, coords_diff[:,A,0], theta[:,A], phi_y)
                    L[A][2] -= oe.contract('r,rm,r,r,rn->mn', weights, phi, coords_diff[:,A,1], theta[:,A], phi_x)
                    J[A][0] -= 0.5 * (L[A][0] - L[A][0].T)
                    J[A][1] -= 0.5 * (L[A][1] - L[A][1].T)
                    J[A][2] -= 0.5 * (L[A][2] - L[A][2].T)

                #for A in range(natom):
                #    for alpha in range(3):
                #        print(J[A][alpha])

                # Build the local average position, R^0.
                R_0_numerator = oe.contract('ab,ac->bc', zeta, R)
                R_0_denominator = oe.contract('ab->b', zeta)
                R_0 = np.zeros((natom,3))
                for B in range(natom):
                    R_0[B] = R_0_numerator[B] / R_0_denominator[B]

                # TEST: Testing the local average position using eq. 40.
                assert np.allclose(oe.contract('ab,ac->bc', zeta, R) - oe.contract('ab,bc->bc', zeta, R_0), 0, atol=1e-12)

                # Building the moment of inertia like tensor, K.
                K = np.zeros((natom,3,3))
                for B in range(natom):
                    X0_TX0 = np.eye(3) * oe.contract('a,a->', R_0[B], R_0[B])
                    X0X0_T = oe.contract('a,b->ab', R_0[B], R_0[B])
                    for A in range(natom):
                        X_TX = np.eye(3) * oe.contract('a,a->', R[A], R[A])
                        XX_T = oe.contract('a,b->ab', R[A], R[A])    
                        K[B] += zeta[A][B] * ((X_TX - X0_TX0) - (XX_T - X0X0_T))
                #print(K)

                # Compute K^-1 J for Gamma". Using np.linalg.lstsq() to accomadate linear molecules.
                #K_inv_J = oe.contract('abc,acmn->abmn', np.linalg.inv(K), J)
                K_inv_J = np.zeros((natom, 3, nbf, nbf))
                for B in range(natom):
                    J_flat = J[B].reshape(3, nbf * nbf)
                    x, _, _, _ = np.linalg.lstsq(K[B], J_flat, rcond=None)
                    K_inv_J[B] = x.reshape(3, nbf, nbf)

                # Computing Gamma".
                self.G2 = np.zeros((natom,3,nbf,nbf))
                for A in range(natom):
                    for B in range(natom):
                        self.G2[A][0] -= zeta[A][B] * ((R[A][1] - R_0[B][1]) * K_inv_J[B][2] - (R[A][2] - R_0[B][2]) * K_inv_J[B][1])
                        self.G2[A][1] -= zeta[A][B] * ((R[A][2] - R_0[B][2]) * K_inv_J[B][0] - (R[A][0] - R_0[B][0]) * K_inv_J[B][2])
                        self.G2[A][2] -= zeta[A][B] * ((R[A][0] - R_0[B][0]) * K_inv_J[B][1] - (R[A][1] - R_0[B][1]) * K_inv_J[B][0])

                # Compute total Gamma.
                self.G = self.G1 + self.G2

                # Building Gamma tilde.
                self.G_tilde = np.zeros((natom,3,nbf,nbf))

                # Compute the diagonalized overlap matrix, s.
                s, U = np.linalg.eigh(self.S)

                # Compute Gamma tilde.
                G_tilde_denominator = np.zeros((nbf,nbf))
                for lambd in range(nbf):
                    for sigma in range(nbf):
                        G_tilde_denominator[lambd][sigma] += (s[lambd] + s[sigma])
                self.G_tilde = 2.0 * oe.contract('ms,absl,ln->abmn', U, oe.contract('lm,abmn,ns->abls', U.T, self.G, U) / G_tilde_denominator, U.T)

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
                #t1 = time.time()
                #print("Time Building Gamma:", t1-t0)


















