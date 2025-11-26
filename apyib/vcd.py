""" Combines all the functions to compute VCD spectra."""

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



class vcd(object):
    """
    Combines the functions for the Hessian, APTs, and AATs to compute a VCD spectrum.
    """
    def __init__(self, parameters):

        # Define the calculation parameters.
        self.parameters = parameters

        # Set the molecule.
        self.molecule = psi4.geometry(parameters['geom'])
        self.natom = self.molecule.natom()



    def compute_vcd_from_input(self, Hessian, APT, AAT_elec):
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

        # Obtain Hessian from user input.
        hessian = Hessian.copy()

        # Mass weight the Cartesian coordinate Hessian [E_h / (m_e * a_0**2].
        mass_weight = np.eye((3 * self.natom))
        for i in range(3 * self.natom):
            mass_weight[i] *= np.sqrt(1 / (self.molecule.mass(i // 3) * _u / _me))

        # Obtain the normal coordinate Hessian [E_h / (m_e * a_0**2].
        hessian_m = mass_weight.T @ hessian @ mass_weight

        # Compute the eigenvalues and eigenvectors.
        l, L_m = np.linalg.eigh(hessian_m)

        # Mass weight the eigenvectors.
        L = mass_weight @ L_m

        # Flip the eigenvalues and eigenvectors and capture only the vibrational degrees of freedom.
        S = np.flip(L, 1)[:,0:(3 * self.natom - 6)]
        l = np.flip(l)[0:(3 * self.natom - 6)]

        # Compute the vibrational frequencies in atomic units.
        w = np.sqrt(l)

        # Compute the APTs in the normal coordinate basis [(e * a_0) / (a_0 * sqrt(m_e))].
        P = APT.copy()
        P_i = P.T @ S
        #print("Length Gauge Atomic Polar Tensor (Normal Mode Basis):")
        #print(P_i.T)

        # Compute the electronic component of the AATs [(e * h) / m_e].
        I = AAT_elec.copy()

        # Compute the nuclear component of the AATs [(e * h) / m_e].
        geom, mass, elem, Z, uniq = self.molecule.to_arrays()

        J = np.zeros((3 * self.natom, 3)) 
        for lambd_alpha in range(3 * self.natom):
            alpha = lambd_alpha % 3 
            lambd = lambd_alpha // 3
            for beta in range(3):
                for gamma in range(3):
                    J[lambd_alpha][beta] += 1/4 * eps[alpha, beta, gamma] * geom[lambd, gamma] * Z[lambd]

        # Compute the AATs in the normal coordinate basis [(e * h) / m_e].
        M = I + J
        M_i = M.T @ S
        #print("Atomic Axial Tensor (Normal Mode Basis):")
        #print(M_i.T)

        # Compute the VCD rotational strengths and IR dipole strengths.
        R = np.zeros((3 * self.natom - 6)) 
        D = np.zeros((3 * self.natom - 6)) 
        for i in range(3 * self.natom - 6):
            R[i] = oe.contract('i,i->', P_i[:,i].real, M_i[:,i].real)
            D[i] = oe.contract('i,i->', P_i[:,i].real, P_i[:,i].real)

        print("\nFrequency   IR Intensity   Rotational Strength")
        print(" (cm-1)      (km/mol)    (esu**2 cm**2 10**44)")
        print("----------------------------------------------")
        for i in range(3 * self.natom - 6): 
            print(f" {w[i] * conv_freq_au2wavenumber:7.2f}     {D[i] * conv_ir_au2kmmol:8.3f}        {R[i] * conv_vcd_au2cgs:8.3f}")

        return w * conv_freq_au2wavenumber, D * conv_ir_au2kmmol, R * conv_vcd_au2cgs



    def compute_LGOI_vcd_from_input(self, Hessian, APT_LG, APT_VG, AAT_elec):
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

        # Obtain Hessian from user input.
        hessian = Hessian.copy()

        # Mass weight the Cartesian coordinate Hessian [E_h / (m_e * a_0**2].
        mass_weight = np.eye((3 * self.natom))
        for i in range(3 * self.natom):
            mass_weight[i] *= np.sqrt(1 / (self.molecule.mass(i // 3) * _u / _me))

        # Obtain the normal coordinate Hessian [E_h / (m_e * a_0**2].
        hessian_m = mass_weight.T @ hessian @ mass_weight

        # Compute the eigenvalues and eigenvectors.
        l, L_m = np.linalg.eigh(hessian_m)

        # Mass weight the eigenvectors.
        L = mass_weight @ L_m 

        # Flip the eigenvalues and eigenvectors and capture only the vibrational degrees of freedom.
        S = np.flip(L, 1)[:,0:(3 * self.natom - 6)] 
        l = np.flip(l)[0:(3 * self.natom - 6)] 

        # Compute the vibrational frequencies in atomic units.
        w = np.sqrt(l)

        # Compute the length gauge APTs in the normal coordinate basis [(e * a_0) / (a_0 * sqrt(m_e))].
        P_r = APT_LG.copy()
        P_ri = P_r.T @ S 
        #print("Length Gauge Atomic Polar Tensor (Normal Mode Basis):")
        #print(P_ri.T)

        # Compute velocity gauge APTs.
        P_p = APT_VG.copy()
        P_pi = P_p.T @ S 
        #print("Velocity Gauge Atomic Polar Tensor (Normal Mode Basis):")
        #print(P_pi.T)

        # Compute the electronic component of the AATs [(e * h) / m_e].
        I = AAT_elec.copy()

        # Compute the nuclear component of the AATs [(e * h) / m_e].
        geom, mass, elem, Z, uniq = self.molecule.to_arrays()

        J = np.zeros((3 * self.natom, 3)) 
        for lambd_alpha in range(3 * self.natom):
            alpha = lambd_alpha % 3 
            lambd = lambd_alpha // 3
            for beta in range(3):
                for gamma in range(3):
                    J[lambd_alpha][beta] += 1/4 * eps[alpha, beta, gamma] * geom[lambd, gamma] * Z[lambd]

        # Compute the AATs in the normal coordinate basis [(e * h) / m_e].
        M = I + J 
        print("Total AAT (Cartesian Coordinate Basis):")
        print(M)
        M_i = M.T @ S 
        #print("Atomic Axial Tensor (Normal Mode Basis):")
        #print(M_i.T)

        # Compute tensor to SVD.
        R_rl_list = []
        R_pl_list = []
        R_rl_LGOI_list = []
        D_rr_list = []
        D_rp_list = []
        D_pp_list = []
        DoS_list = []
        det_U = []
        det_V = []

        for i in range(3 * self.natom - 6):
            R_rl = np.zeros((3, 3))
            R_pl = np.zeros((3, 3))
            D_rr = np.zeros((3, 3))
            D_rp = np.zeros((3, 3))
            D_pp = np.zeros((3, 3))
 
            # Compute the full rotatory strength and dipole strength tensor for a given normal mode i.
            for a in range(3):
                for b in range(3):
                    R_rl[a][b] = P_ri[a,i].real * M_i[b,i].real
                    R_pl[a][b] = P_pi[a,i].real * M_i[b,i].real
                    D_rr[a][b] = P_ri[a,i].real * P_ri[b,i].real
                    D_rp[a][b] = P_ri[a,i].real * P_pi[b,i].real
                    D_pp[a][b] = P_pi[a,i].real * P_pi[b,i].real

            # Compute the trace of the length gauge and velocity gauge rotatory strengths and length gauge, velocity gauge, and mixed gauge dipole strengths.
            R_rl_list.append(np.trace(R_rl))
            R_pl_list.append(np.trace(R_pl))
            D_rr_list.append(np.trace(D_rr))
            D_rp_list.append(np.trace(D_rp))
            D_pp_list.append(np.trace(D_pp))

            # Compute LGOI rotatory strength.
            U, D_rp_diag, Vt = np.linalg.svd(D_rp)
            det_U.append(np.linalg.det(U))
            det_V.append(np.linalg.det(Vt))

            R_rl_LGOI = U.T @ R_rl @ Vt.T
            R_rl_LGOI_list.append(np.trace(R_rl_LGOI))

            # Compute degree of symmetry.
            A_rp = 0.5 * (D_rp - D_rp.T)
            norm_A_rp = np.linalg.norm(A_rp, ord='fro')
            norm_D_rp = np.linalg.norm(D_rp, ord='fro')
            DoS = 1 - (norm_A_rp / norm_D_rp)
            DoS_list.append(DoS)

        print("\nMode   Frequency              IR Intensity                       Rotational Strength                Analysis")
        print("        (cm-1)                  (km/mol)                        (esu**2 cm**2 10**44)")
        print("                      LG          VG          Mixed           LG          VG          LG(OI)      DoS   det(U)  det(V)")
        print("----------------------------------------------------------------------------------------------------------------------------")
        for i in range(3 * self.natom - 6): 
            print(f"{3 * self.natom - 6 - i}      {w[i] * conv_freq_au2wavenumber:7.2f}    {D_rr_list[i] * conv_ir_au2kmmol:8.3f}     {D_pp_list[i] * conv_ir_au2kmmol:8.3f}     {D_rp_list[i] * conv_ir_au2kmmol:8.3f}     {R_rl_list[i] * conv_vcd_au2cgs:8.3f}     {R_pl_list[i] * conv_vcd_au2cgs:8.3f}     {R_rl_LGOI_list[i] * conv_vcd_au2cgs:8.3f}   {DoS_list[i]:8.3f} {det_U[i]:8.3f} {det_V[i]:8.3f}")

        D_rr_list = np.array(D_rr_list)
        D_pp_list = np.array(D_pp_list)
        R_rl_list = np.array(R_rl_list)
        R_pl_list = np.array(R_pl_list)
        R_rl_LGOI_list = np.array(R_rl_LGOI_list)

        return w * conv_freq_au2wavenumber, D_rr_list * conv_ir_au2kmmol, D_pp_list * conv_ir_au2kmmol, R_rl_list * conv_vcd_au2cgs, R_pl_list * conv_vcd_au2cgs, R_rl_LGOI_list * conv_vcd_au2cgs









