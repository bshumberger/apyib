""" Combines all the functions to compute VCD spectra."""

import psi4
import numpy as np
import scipy.linalg as la
from apyib.utils import run_psi4
from apyib.utils import compute_F_SO
from apyib.utils import compute_ERI_SO
from apyib.hamiltonian import Hamiltonian
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn
from apyib.finite_difference import finite_difference
from apyib.aats import AAT 
from apyib.aats import compute_mo_overlap
from apyib.aats import compute_phase



class vcd(object):
    """
    Combines the functions for the Hessian, APTs, and AATs to compute a VCD spectrum.
    """
    def __init__(self, parameters):

        # Define the calculation parameters.
        self.parameters = parameters

        # Set the Hamiltonian and perform a standard Hartree-Fock calculation.
        self.H = Hamiltonian(self.parameters)
        self.wfn = hf_wfn(self.H)
        self.E_SCF, self.E_tot, self.C = self.wfn.solve_SCF(self.parameters)
        print("Hartree-Fock Electronic Energy: ", self.E_SCF)
        print("Total Energy: ", self.E_tot, "\n")
        self.unperturbed_basis = self.H.basis_set
        self.unperturbed_wfn = self.C

        # Set the number of atoms.
        self.natom = self.H.molecule.natom()

        # Perform calculations for the chosen method if not Hartree-Fock.
        if self.parameters['method'] == 'RHF':
            self.unperturbed_t2 = 0
        if self.parameters['method'] == 'MP2':
            self.wfn_MP2 = mp2_wfn(self.parameters, self.E_SCF, self.E_tot, self.C)
            self.E_MP2, self.t2 = self.wfn_MP2.solve_MP2()
            self.unperturbed_t2 = self.t2
        if self.parameters['method'] == 'CID':
            self.wfn_cid = ci_wfn(self.parameters, self.E_SCF, self.E_tot, self.C)
            self.E_CID, self.t2 = self.wfn_cid.solve_CID()
            self.unperturbed_t2 = self.t2

    def compute_vcd(self, hessian_nuc_disp, APT_nuc_disp, APT_elec_field_pert, AAT_nuc_disp, AAT_mag_field_pert):
        """
        Compute components for VCD spectral generation.
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

        # Initializing finite difference object.
        fin_diff = finite_difference(self.parameters, self.unperturbed_basis, self.unperturbed_wfn)
 
        # Compute gradients for finite difference calculations of the Hessian.
        pos_e, neg_e = fin_diff.second_nuclear_displacements(hessian_nuc_disp)

        # Compute the Hessian [E_h / a_0**2].
        pos_e = np.array(pos_e)
        neg_e = np.array(neg_e)
        pos_e = pos_e.reshape((3 * self.natom, 3 * self.natom))
        neg_e = neg_e.reshape((3 * self.natom, 3 * self.natom))
        hessian = (pos_e - neg_e) / (2 * hessian_nuc_disp)

        # Mass weight the Cartesian coordinate Hessian [E_h / (m_e * a_0**2].
        mass_weight = np.eye((3 * self.natom))
        for i in range(3 * self.natom):
            mass_weight[i] *= np.sqrt(1 / (self.H.molecule.mass(i // 3) * _u / _me))

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

        # Compute dipoles for finite difference calculations of the APTs.
        pos_mu, neg_mu = fin_diff.nuclear_and_electric_field_perturbations(APT_nuc_disp, APT_elec_field_pert)

        # Compute the APTs in the normal coordinate basis [(e * a_0) / (a_0 * sqrt(m_e))].
        pos_mu = np.array(pos_mu)
        neg_mu = np.array(neg_mu)
        pos_mu = pos_mu.reshape((3 * self.natom, 3))
        neg_mu = neg_mu.reshape((3 * self.natom, 3))
        P = (pos_mu - neg_mu) / (2 * APT_nuc_disp)
        P_i = P.T @ S

        # Compute energies and wavefunctions for finite difference calculations of nuclear displacements for AATs.
        nuc_pos_e, nuc_neg_e, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, nuc_pos_t2, nuc_neg_t2 = fin_diff.nuclear_displacements(AAT_nuc_disp)

        # Compute energies and wavefunctions for finite difference calculations of magnetic field perturbations for AATs.
        mag_pos_e, mag_neg_e, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis, mag_pos_t2, mag_neg_t2 = fin_diff.magnetic_field_perturbations(AAT_mag_field_pert)

        # Set up the AAT object.
        AATs = AAT(self.parameters, self.wfn.nbf, self.wfn.ndocc, self.unperturbed_wfn, self.unperturbed_basis, self.unperturbed_t2, nuc_pos_wfns, nuc_neg_wfns, nuc_pos_basis, nuc_neg_basis, nuc_pos_t2, nuc_neg_t2, mag_pos_wfns, mag_neg_wfns, mag_pos_basis, mag_neg_basis, mag_pos_t2, mag_neg_t2, AAT_nuc_disp, AAT_mag_field_pert)

        # Compute the electronic component of the AATs [(e * h) / m_e].
        I = np.zeros((3 * self.natom, 3), dtype=np.cdouble)
        for lambd_alpha in range(3 * self.natom):
            for beta in range(3):
                if self.parameters['method'] == 'RHF':
                    I[lambd_alpha][beta] = AATs.compute_hf_aat(lambd_alpha, beta)
                elif self.parameters['method'] == 'MP2' or self.parameters['method'] == 'CID':
                    I[lambd_alpha][beta] = AATs.compute_cid_aat(lambd_alpha, beta)

        # Compute the nuclear component of the AATs [(e * h) / m_e].
        geom, mass, elem, Z, uniq = self.H.molecule.to_arrays()

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

        # Compute the VCD rotational strengths and IR dipole strengths.
        R = np.zeros((3 * self.natom - 6))
        D = np.zeros((3 * self.natom - 6))
        for i in range(3 * self.natom - 6):
            R[i] = np.einsum('i,i->', P_i[:,i].real, M_i[:,i].real)
            D[i] = np.einsum('i,i->', P_i[:,i].real, P_i[:,i].real)

        print("\nFrequency   IR Intensity   Rotational Strength")
        print(" (cm-1)      (km/mol)    (esu**2 cm**2 10**44)")
        print("----------------------------------------------")
        for i in range(3 * self.natom - 6):
            print(f" {w[i] * conv_freq_au2wavenumber:7.2f}     {D[i] * conv_ir_au2kmmol:8.3f}        {R[i] * conv_vcd_au2cgs:8.3f}")

        return w * conv_freq_au2wavenumber, D * conv_ir_au2kmmol, R * conv_vcd_au2cgs










