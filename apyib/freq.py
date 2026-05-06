""" Combines all the functions to compute frequencies."""

import psi4
import numpy as np
import opt_einsum as oe
import scipy.linalg as la
from apyib.utils import run_psi4
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn



class frequency(object):
    """
    Combines the functions for the momentum Hessian and position Hessian for calculations of phase-space vibrational frequencies.
    """
    def __init__(self, parameters):

        # Define the calculation parameters.
        self.parameters = parameters

        # Set the molecule.
        self.molecule = psi4.geometry(parameters['geom'])
        self.natom = self.molecule.natom()



    def compute_freq_from_input(self, position_Hessian, momentum_Hessian=None):
        """
        Compute components for VCD spectral generation from user input.
        """
        # Set up physical constants.
        _c = psi4.qcel.constants.get("speed of light in vacuum") # m/s
        _me = psi4.qcel.constants.get("electron mass") # kg
        _na = psi4.qcel.constants.get("Avogadro constant") # 1/mol
        _e = psi4.qcel.constants.get("atomic unit of charge") # C 
        _e0 = psi4.qcel.constants.get("electric constant") # F/m = s^4 A^2/(kg m^3)
        _h = psi4.qcel.constants.get("Planck constant") # J s
        _hbar = _h/(2*np.pi) # J s/rad
        _mu0 = 1/(_c**2 * _e0) # s^2/(m F) = kg m/(s^2 A^2)
        _ke = 1/(4 * np.pi * _e0) # kg m^3/(C^2 s^2)
        _alpha = _ke * _e**2/(_hbar * _c) # dimensionless
        _a0 = _hbar/(_me * _c * _alpha) # m 
        _Eh = (_hbar**2)/(_me * _a0**2) # J 
        _u = 1/(1000 * _na) # kg
        _D = 1/(10**(21) * _c) # C m/Debye
        _bohr2angstroms = _a0 * 10**(10)

        # Frequencies in au --> cm^-1
        conv_freq_au2wavenumber = np.sqrt(_Eh/(_a0 * _a0 * _me)) * (1.0/(2.0 * np.pi * _c * 100.0))
        # IR intensities in au --> km/mol
        conv_ir_au2kmmol = (_e**2 * _ke * _na * np.pi)/(1000 * 3 * _me * _c**2)
        # VCD rotatory strengths in au --> (esu cm)**2 * (10**(44))
        conv_vcd_au2cgs = (_e**2 * _hbar * _a0)/(_me * _c) * (1000 * _c)**2 * (10**(44))

        # Set up the Levi-Civita tensor.
        epsilon = [[[0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]],
                   [[0, 0, -1],
                    [0, 0, 0,],
                    [1, 0, 0]],
                   [[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]]]

        eps = np.array(epsilon)

        # Compute the molecules center of mass.
        R, M, elem, Z, uniq = self.molecule.to_arrays()
        R = np.array(R)
        M = np.array(M)
        M_tot = np.sum(M)

        COM = oe.contract('ab,a->b', R, M) / oe.contract('a->', M)

        # Compute center of mass shifted coordinates.
        R_COM = np.zeros_like(R)
        for N in range(self.natom):
            R_COM[N] += R[N] - COM

        # Compute the moment of intertia tensor.
        I = np.zeros((3,3))
        I_int1 = oe.contract('a,ab,ab->', M, R_COM, R_COM)
        I_int2 = - oe.contract('a,ab,ac->bc', M, R_COM, R_COM)
        for alpha in range(3):
            I[alpha][alpha] += I_int1
        I += I_int2
        I_p, X = np.linalg.eigh(I)

        # Build the D matrix to separate out rotations and translations.
        # Translation vectors.
        D_all = np.zeros((6, 3 * self.natom))
        for N in range(self.natom):
            for alpha in range(3):
                D_all[alpha][3 * N + alpha] += np.sqrt(M[N])

        # Rotation vectors (up to 3 vectors; Ochterski Eq. 5).
        P = oe.contract('ab,bc->ac', R_COM, X)
        for N in range(self.natom):
            for alpha in range(3):
                D_all[3][3 * N + alpha] += (P[N][1] * X[alpha][2] - P[N][2] * X[alpha][1]) * np.sqrt(M[N])
                D_all[4][3 * N + alpha] += (P[N][2] * X[alpha][0] - P[N][0] * X[alpha][2]) * np.sqrt(M[N])
                D_all[5][3 * N + alpha] += (P[N][0] * X[alpha][1] - P[N][1] * X[alpha][0]) * np.sqrt(M[N])

        # Normalize each vector and discard any with zero norm. For linear molecules, one rotation vector 
        # will have zero norm and must be removed (Ochterski Section 2.3).
        D_clean = []
        for k in range(6):
            n = np.linalg.norm(D_all[k])
            if n > 1e-10:
                D_clean.append(D_all[k] / n)
        D = np.array(D_clean)  # (n_tr, 3*natom), n_tr = 5 (linear) or 6 (nonlinear)
        n_tr = D.shape[0]

        # Using SVD to find the transformation matrix which takes us from mass weighted Cartesian coordinates to internal coordinates.
        U, s, V = np.linalg.svd(D.T, full_matrices=True)
        B = U[:,n_tr:]

        # Mass-weighted coordinates: q_i = r_i * sqrt(m_i).
        R_COM_mw = np.zeros_like(R_COM)
        for N in range(self.natom):
            R_COM_mw[N] = R_COM[N] * np.sqrt(M[N])

        # Diagonalize and build I^{-1} in the original frame.
        I_inv = np.linalg.inv(I)

        # Build the Eckart projector: P = I - P_trans - P_rot.
        #
        # P_trans removes translations:
        #   P_trans[i,j] = sqrt(m_i * m_j) / M_total * delta(icart, jcart)
        #
        # P_rot removes rotations using the inverse inertia tensor:
        #   P_rot[i,j] = sum_{a,b,c,d} eps(a,b,icart) * q[iatom,b]
        #                * I^{-1}[a,c] * eps(c,d,jcart) * q[jatom,d]
        P = np.eye(3 * self.natom)
        for i in range(3 * self.natom):
            icart = i % 3
            iatom = i // 3
            imass = M[iatom]
            for j in range(3 * self.natom):
                jcart = j % 3
                jatom = j // 3
                jmass = M[jatom]

                # Translation projection.
                P[i][j] -= np.sqrt(imass * jmass) / M_tot * (1 if icart == jcart else 0)

                # Rotation projection.
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            for d in range(3):
                                P[i][j] -= eps[a][b][icart] * R_COM_mw[iatom][b] * I_inv[a][c] * eps[c][d][jcart] * R_COM_mw[jatom][d]

        # Off setting testing for phase-space momentum Hessian.
        PS = self.parameters.get('hamiltonian', None)
        if PS == 'phase-space':

            # Computing delta (the effective masses) and the unitary transformation matrix diagonalizing the momentum Hessian.
            delta_inv, U = np.linalg.eigh(momentum_Hessian)
            delta = 1.0 / delta_inv * (_me / _u) 
            print("Effective Masses:")
            print(delta)

            # Compute phase-space vibrational frequencies.
            delta_half = np.eye(3 * self.natom)
            for i in range(3 * self.natom):
                delta_half[i] *= np.sqrt(delta_inv[i])

            K_eff = delta_half @ U.T @ position_Hessian @ U @ delta_half
            h_mw_eig, L_mw = np.linalg.eigh(K_eff)
            h_mw_eig = np.flip(h_mw_eig)
            w = np.where(h_mw_eig >= 0, np.sqrt(h_mw_eig), -np.sqrt(-h_mw_eig))
            L = U @ delta_half @ L_mw
            S = np.flip(L, 1)[:,0:(3 * self.natom - 6)]
            #print("Normal Coordinate Hessian:")
            #print(K_eff)
            #print("Normal Coordinate Hessian Eigenvalues:")
            #print(h_mw_eig)

            # Obtain the internal coordinate Hessian.
            hessian_int = B.T @ K_eff @ B
            h_int_eig, L_int = np.linalg.eigh(hessian_int)
            h_int_eig = np.flip(h_int_eig)
            w_int = np.where(h_int_eig >= 0, np.sqrt(h_int_eig), -np.sqrt(-h_int_eig))
            L1 = (U @ delta_half)[:,0:(3 * self.natom - 6)] @ L_int
            S1 = np.flip(L1, 1)[:,0:(3 * self.natom - 6)]
            #print("Internal Coordinate Hessian:")
            #print(hessian_int)
            #print("Internal Coordinate Hessian Eigenvalues:")
            #print(h_int_eig)

            # Obtain the projected internal coordinate Hessian.
            hessian_proj = P @ K_eff @ P
            h_proj_eig, L_proj = np.linalg.eigh(hessian_proj)
            h_proj_eig = np.flip(h_proj_eig)
            w_proj = np.where(h_proj_eig >= 0, np.sqrt(h_proj_eig), -np.sqrt(-h_proj_eig))
            L2 = U @ delta_half @ L_proj
            S2 = np.flip(L2, 1)[:,0:(3 * self.natom - 6)]
            #print("Projected Internal Coordinate Hessian:")
            #print(hessian_proj)
            #print("Projected Internal Coordinate Hessian Eigenvalues:")
            #print(h_proj_eig)

            print("\nNormal Coordinate     Internal Coordinate    Projected Internal Coordinate")
            print("    Frequency           Frequency (Gaussian)       Frequency (Crawford)")
            print("      (cm-1)                  (cm-1)                    (cm-1)")
            print("---------------------------------------------------------------------")
            for i in range(3 * self.natom):
                if i >= len(w_int):
                    print(f"   {w[i] * conv_freq_au2wavenumber:10.2f} {'N/A':>23} {w_proj[i] * conv_freq_au2wavenumber:25.2f}")
                else:
                    print(f"   {w[i] * conv_freq_au2wavenumber:10.2f} {w_int[i] * conv_freq_au2wavenumber:23.2f} {w_proj[i] * conv_freq_au2wavenumber:25.2f}")

            return w_proj * conv_freq_au2wavenumber, S2

        if PS == None:
            # Mass weight the Cartesian coordinate Hessian [E_h / (m_e * a_0**2].
            mass_weight = np.eye((3 * self.natom))
            for i in range(3 * self.natom):
                mass_weight[i] *= np.sqrt(1 / (self.molecule.mass(i // 3) * _u / _me))

            n_vib = 3 * self.natom - n_tr

            # Obtain the normal coordinate Hessian [E_h / (m_e * a_0**2].
            hessian_mw = mass_weight.T @ position_Hessian @ mass_weight
            h_mw_eig, L_mw = np.linalg.eigh(hessian_mw)
            h_mw_eig = np.flip(h_mw_eig)
            w_m = np.where(h_mw_eig >= 0, np.sqrt(h_mw_eig), -np.sqrt(-h_mw_eig))
            L = mass_weight @ L_mw
            S = np.flip(L, 1)[:,0:(3 * self.natom - 6)]
            #print("Normal Coordinate Hessian:")
            #print(hessian_mw)
            #print("Normal Coordinate Hessian Eigenvalues:")
            #print(h_mw_eig)

            # Obtain the internal coordinate Hessian (Gaussian).
            hessian_int = B.T @ hessian_mw @ B
            h_int_eig, L_int = np.linalg.eigh(hessian_int)
            h_int_eig = np.flip(h_int_eig)
            w_int = np.where(h_int_eig >= 0, np.sqrt(h_int_eig), -np.sqrt(-h_int_eig))
            L1 = mass_weight[:,0:(3 * self.natom - 6)] @ L_int
            S1 = np.flip(L1, 1)[:,0:(3 * self.natom - 6)] 
            #print("Internal Coordinate Hessian:")
            #print(hessian_int)
            #print("Internal Coordinate Hessian Eigenvalues:")
            #print(h_int_eig)

            # Obtain the projected internal coordinate Hessian (Crawford).
            hessian_proj = P @ hessian_mw @ P
            h_proj_eig, L_proj = np.linalg.eigh(hessian_proj)
            h_proj_eig = np.flip(h_proj_eig)
            w_proj = np.where(h_proj_eig >= 0, np.sqrt(h_proj_eig), -np.sqrt(-h_proj_eig))
            L2 = mass_weight @ L_proj
            S2 = np.flip(L2, 1)[:,0:(3 * self.natom - 6)] 
            #print("Projected Internal Coordinate Hessian:")
            #print(hessian_proj)
            #print("Projected Internal Coordinate Hessian Eigenvalues:")
            #print(h_proj_eig)

            print("\nNormal Coordinate     Internal Coordinate    Projected Internal Coordinate")
            print("    Frequency           Frequency (Gaussian)       Frequency (Crawford)")
            print("      (cm-1)                  (cm-1)                    (cm-1)")
            print("---------------------------------------------------------------------")
            for i in range(3 * self.natom):
                if i >= len(w_int):
                    print(f"   {w_m[i] * conv_freq_au2wavenumber:10.2f} {'N/A':>23} {w_proj[i] * conv_freq_au2wavenumber:25.2f}")
                else:
                    print(f"   {w_m[i] * conv_freq_au2wavenumber:10.2f} {w_int[i] * conv_freq_au2wavenumber:23.2f} {w_proj[i] * conv_freq_au2wavenumber:25.2f}")

            return w_proj * conv_freq_au2wavenumber, S2








