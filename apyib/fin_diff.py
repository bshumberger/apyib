"""Contains streamlined functions for finite difference procedures."""

import numpy as np
import psi4
import opt_einsum as oe
from apyib.energy import energy, phase_corrected_energy
from apyib.utils import total_energy


class finite_difference(object):
    """Finite difference driver for numerical Hessian, APT, and AAT computations."""

    def __init__(self, parameters, unperturbed_basis, unperturbed_C):
        self.parameters = parameters
        self.molecule = psi4.geometry(self.parameters['geom'])
        self.geom = self.molecule.geometry().np
        self.natom = self.molecule.natom()
        self.unperturbed_basis = unperturbed_basis
        self.unperturbed_C = unperturbed_C

    def compute_Hessian(self, nuc_pert_strength):
        """Compute the nuclear Hessian by central finite difference.

        Parameters
        ----------
        nuc_pert_strength : float
            Step size for nuclear coordinate displacements (Bohr).

        Returns
        -------
        hessian : ndarray, shape (3*natom, 3*natom)
            Second derivative of total energy w.r.t. nuclear coordinates [E_h / a_0^2].
        """
        # Hessian via double central difference:
        #   H[alpha, beta] = (dE/dbeta at +alpha  -  dE/dbeta at -alpha) / 2h
        # where each inner gradient uses a central difference in beta.
        # alpha and beta index the 3N Cartesian coordinates; alpha//3 is the
        # atom index and alpha%3 is the x/y/z component.
        n = 3 * self.natom
        h = nuc_pert_strength
        # pos_E[alpha, beta] = dE/dbeta evaluated at +alpha displacement
        # neg_E[alpha, beta] = dE/dbeta evaluated at -alpha displacement
        pos_E = np.zeros((n, n))
        neg_E = np.zeros((n, n))

        # Outer loop: positive alpha displacements.
        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] += h
            pert_geom_alpha = np.copy(pert_geom)

            pos_e = np.zeros(n)   # E(+alpha, +beta)
            neg_e = np.zeros(n)   # E(+alpha, -beta)

            for beta in range(n):
                pert_geom_alpha[beta // 3][beta % 3] += h
                self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom_alpha))
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                E_list, _, _, _ = energy(self.parameters)
                pos_e[beta] = total_energy(E_list)
                pert_geom_alpha = np.copy(pert_geom)

            for beta in range(n):
                pert_geom_alpha[beta // 3][beta % 3] -= h
                self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom_alpha))
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                E_list, _, _, _ = energy(self.parameters)
                neg_e[beta] = total_energy(E_list)
                pert_geom_alpha = np.copy(pert_geom)

            # Central difference in beta gives the gradient at +alpha.
            pos_E[alpha] = (pos_e - neg_e) / (2 * h)
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Outer loop: negative alpha displacements.
        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] -= h
            pert_geom_alpha = np.copy(pert_geom)

            pos_e = np.zeros(n)   # E(-alpha, +beta)
            neg_e = np.zeros(n)   # E(-alpha, -beta)

            for beta in range(n):
                pert_geom_alpha[beta // 3][beta % 3] += h
                self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom_alpha))
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                E_list, _, _, _ = energy(self.parameters)
                pos_e[beta] = total_energy(E_list)
                pert_geom_alpha = np.copy(pert_geom)

            for beta in range(n):
                pert_geom_alpha[beta // 3][beta % 3] -= h
                self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom_alpha))
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                E_list, _, _, _ = energy(self.parameters)
                neg_e[beta] = total_energy(E_list)
                pert_geom_alpha = np.copy(pert_geom)

            # Central difference in beta gives the gradient at -alpha.
            neg_E[alpha] = (pos_e - neg_e) / (2 * h)
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Final central difference in alpha gives the Hessian.
        return (pos_E - neg_E) / (2 * h)

    def compute_APT(self, nuc_pert_strength, elec_pert_strength):
        """Compute the atomic polar tensor by central finite difference.

        Parameters
        ----------
        nuc_pert_strength : float
            Step size for nuclear coordinate displacements (Bohr).
        elec_pert_strength : float
            Step size for electric field perturbations (a.u.).

        Returns
        -------
        P : ndarray, shape (3*natom, 3)
            Atomic polar tensor.
        """
        # APT via mixed nuclear/electric central difference:
        #   P[alpha, beta] = -d^2E / (dR_alpha dF_beta)
        # The electric dipole mu_beta = -dE/dF_beta is computed by central
        # difference in the field at each nuclear displacement, then the
        # nuclear central difference gives the APT.
        n = 3 * self.natom
        h_nuc = nuc_pert_strength
        h_elec = elec_pert_strength
        # pos_mu[alpha, beta] = mu_beta at +alpha nuclear displacement
        # neg_mu[alpha, beta] = mu_beta at -alpha nuclear displacement
        pos_mu = np.zeros((n, 3))
        neg_mu = np.zeros((n, 3))

        # Positive nuclear displacements.
        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] += h_nuc
            self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            pos_e = np.zeros(3)   # E(+alpha, +F_beta)
            neg_e = np.zeros(3)   # E(+alpha, -F_beta)

            for beta in range(3):
                self.parameters['F_el'][beta] += h_elec
                E_list, _, _, _ = energy(self.parameters)
                pos_e[beta] = total_energy(E_list)
                self.parameters['F_el'][beta] -= h_elec

            for beta in range(3):
                self.parameters['F_el'][beta] -= h_elec
                E_list, _, _, _ = energy(self.parameters)
                neg_e[beta] = total_energy(E_list)
                self.parameters['F_el'][beta] += h_elec

            # mu = -dE/dF (minus sign: dipole from energy derivative)
            pos_mu[alpha] = -(pos_e - neg_e) / (2 * h_elec)
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Negative nuclear displacements.
        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] -= h_nuc
            self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            pos_e = np.zeros(3)   # E(-alpha, +F_beta)
            neg_e = np.zeros(3)   # E(-alpha, -F_beta)

            for beta in range(3):
                self.parameters['F_el'][beta] += h_elec
                E_list, _, _, _ = energy(self.parameters)
                pos_e[beta] = total_energy(E_list)
                self.parameters['F_el'][beta] -= h_elec

            for beta in range(3):
                self.parameters['F_el'][beta] -= h_elec
                E_list, _, _, _ = energy(self.parameters)
                neg_e[beta] = total_energy(E_list)
                self.parameters['F_el'][beta] += h_elec

            neg_mu[alpha] = -(pos_e - neg_e) / (2 * h_elec)
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Central difference in nuclear coordinate gives the APT.
        return (pos_mu - neg_mu) / (2 * h_nuc)

    def compute_AAT(self, nuc_pert_strength, mag_pert_strength):
        """Compute displaced wavefunctions for AAT finite-difference evaluation.

        Parameters
        ----------
        nuc_pert_strength : float
            Step size for nuclear coordinate displacements (Bohr).
        mag_pert_strength : float
            Step size for magnetic field perturbations (a.u.).

        Returns
        -------
        Twelve lists of displaced MO coefficients, amplitude lists, and basis sets
        (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
        mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T).
        """
        # AAT via mixed nuclear/magnetic central difference:
        #   I[alpha, beta] = Im{ d<Psi|d/dB_beta|Psi> / dR_alpha }
        # The displaced wavefunctions (C, T amplitudes, basis) are collected
        # here so that the AAT tensor contractions in aats.py can be
        # parallelized over (alpha, beta) pairs (see parallel.py).
        # phase_corrected_energy aligns MO phase across displaced geometries.
        n = 3 * self.natom
        nuc_pos_C = [None] * n       # MO coefficients at +R_alpha
        nuc_neg_C = [None] * n       # MO coefficients at -R_alpha
        nuc_pos_basis = [None] * n   # Psi4 BasisSet at +R_alpha
        nuc_neg_basis = [None] * n   # Psi4 BasisSet at -R_alpha
        nuc_pos_T = [None] * n       # CI/MP2 amplitudes at +R_alpha
        nuc_neg_T = [None] * n       # CI/MP2 amplitudes at -R_alpha
        mag_pos_C = [None] * 3       # MO coefficients at +B_beta
        mag_neg_C = [None] * 3       # MO coefficients at -B_beta
        mag_pos_basis = [None] * 3
        mag_neg_basis = [None] * 3
        mag_pos_T = [None] * 3
        mag_neg_T = [None] * 3

        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] += nuc_pert_strength
            self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            _, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            nuc_pos_C[alpha] = C
            nuc_pos_T[alpha] = T_list
            nuc_pos_basis[alpha] = basis
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] -= nuc_pert_strength
            self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            _, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            nuc_neg_C[alpha] = C
            nuc_neg_T[alpha] = T_list
            nuc_neg_basis[alpha] = basis
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        for beta in range(3):
            self.parameters['F_mag'][beta] += mag_pert_strength
            _, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            mag_pos_C[beta] = C
            mag_pos_T[beta] = T_list
            mag_pos_basis[beta] = basis
            self.parameters['F_mag'][beta] -= mag_pert_strength

        for beta in range(3):
            self.parameters['F_mag'][beta] -= mag_pert_strength
            _, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            mag_neg_C[beta] = C
            mag_neg_T[beta] = T_list
            mag_neg_basis[beta] = basis
            self.parameters['F_mag'][beta] += mag_pert_strength

        return (nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T,
                mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T)

    def compute_Nuclear_Gradient(self, nuc_pert_strength):
        """Compute the nuclear energy gradient by central finite difference.

        Parameters
        ----------
        nuc_pert_strength : float
            Step size for nuclear coordinate displacements (Bohr).

        Returns
        -------
        gradient : ndarray, shape (natom, 3)
            First derivative of total energy w.r.t. nuclear coordinates [E_h / a_0].
        nuc_pos_C, nuc_neg_C : list of ndarray
            Displaced MO coefficient matrices.
        nuc_pos_basis, nuc_neg_basis : list
            Displaced Psi4 basis set objects.
        nuc_pos_T, nuc_neg_T : list
            Displaced amplitude lists.
        """
        n = 3 * self.natom
        nuc_pos_E = np.zeros(n)
        nuc_neg_E = np.zeros(n)
        nuc_pos_C = [None] * n
        nuc_neg_C = [None] * n
        nuc_pos_basis = [None] * n
        nuc_neg_basis = [None] * n
        nuc_pos_T = [None] * n
        nuc_neg_T = [None] * n

        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] += nuc_pert_strength
            self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            E_list, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            nuc_pos_C[alpha] = C
            nuc_pos_T[alpha] = T_list
            nuc_pos_basis[alpha] = basis
            nuc_pos_E[alpha] = total_energy(E_list)
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        for alpha in range(n):
            pert_geom = np.copy(self.geom)
            pert_geom[alpha // 3][alpha % 3] -= nuc_pert_strength
            self.molecule.set_geometry(psi4.core.Matrix.from_array(pert_geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            E_list, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            nuc_neg_C[alpha] = C
            nuc_neg_T[alpha] = T_list
            nuc_neg_basis[alpha] = basis
            nuc_neg_E[alpha] = total_energy(E_list)
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        gradient = ((nuc_pos_E - nuc_neg_E) / (2 * nuc_pert_strength)).reshape(self.natom, 3)
        return gradient, nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T

    def compute_Magnetic_Field_Gradient(self, mag_pert_strength):
        """Compute the magnetic field energy gradient by central finite difference.

        Parameters
        ----------
        mag_pert_strength : float
            Step size for magnetic field perturbations (a.u.).

        Returns
        -------
        gradient : ndarray, shape (3,)
        mag_pos_C, mag_neg_C : list of ndarray
        mag_pos_basis, mag_neg_basis : list
        mag_pos_T, mag_neg_T : list
        """
        mag_pos_E = np.zeros(3)
        mag_neg_E = np.zeros(3)
        mag_pos_C = [None] * 3
        mag_neg_C = [None] * 3
        mag_pos_basis = [None] * 3
        mag_neg_basis = [None] * 3
        mag_pos_T = [None] * 3
        mag_neg_T = [None] * 3

        for beta in range(3):
            self.parameters['F_mag'][beta] += mag_pert_strength
            E_list, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            mag_pos_C[beta] = C
            mag_pos_T[beta] = T_list
            mag_pos_basis[beta] = basis
            mag_pos_E[beta] = total_energy(E_list)
            self.parameters['F_mag'][beta] -= mag_pert_strength

        for beta in range(3):
            self.parameters['F_mag'][beta] -= mag_pert_strength
            E_list, T_list, C, basis = phase_corrected_energy(
                self.parameters, self.unperturbed_basis, self.unperturbed_C)
            mag_neg_C[beta] = C
            mag_neg_T[beta] = T_list
            mag_neg_basis[beta] = basis
            mag_neg_E[beta] = total_energy(E_list)
            self.parameters['F_mag'][beta] += mag_pert_strength

        gradient = (mag_pos_E - mag_neg_E) / (2 * mag_pert_strength)
        return gradient, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T

    def compute_momentum_Hessian(self, mom_pert_strength):
        # Make sure you have specified a phase-space Hamiltonian.
        assert self.parameters.get('hamiltonian', None) == 'phase-space'

        # Add the nuclear momentum matrix if it isn't already in the dictionary.
        if self.parameters.get('P_nuc', None) == None:
            self.parameters['P_nuc'] = [list(range(3)) for i in range(self.natom)]
            for A in range(self.natom):
                for alpha in range(3):
                    self.parameters['P_nuc'][A][alpha] = 0.0

        # Set properties of the finite difference procedure.
        pos_E = []
        neg_E = []

        # Computing energies and wavefunctions with positive momentum perturbations.
        for alpha in range(3*self.natom):
            # Perturb the momentum.
            self.parameters['P_nuc'][alpha // 3][alpha % 3] += mom_pert_strength

            pos_e = []
            neg_e = []

            # Perturb the momentum with another positive displacement.
            for beta in range(3*self.natom):
                self.parameters['P_nuc'][beta // 3][beta % 3] += mom_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = total_energy(E_list)

                # Append new energies.
                pos_e.append(E_tot)

                # Reset the second momentum perturbation.
                self.parameters['P_nuc'][beta // 3][beta % 3] -= mom_pert_strength

            # Perturb the momentum with a negative displacement.
            for beta in range(3*self.natom):
                self.parameters['P_nuc'][beta // 3][beta % 3] -= mom_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = total_energy(E_list)

                # Append new energies.
                neg_e.append(E_tot)

                # Reset the second momentum perturbation.
                self.parameters['P_nuc'][beta // 3][beta % 3] += mom_pert_strength

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                g = (pos_e[beta] - neg_e[beta]) / (2 * mom_pert_strength)
                pos_E.append(g)

            # Reset the first momentum perturbation.
            self.parameters['P_nuc'][alpha // 3][alpha % 3] -= mom_pert_strength

        # Computing energies and wavefunctions with negative momentum perturbations.
        for alpha in range(3*self.natom):
            # Perturb the momentum.
            self.parameters['P_nuc'][alpha // 3][alpha % 3] -= mom_pert_strength

            pos_e = []
            neg_e = []

            # Perturb the momentum with another positive displacement.
            for beta in range(3*self.natom):
                self.parameters['P_nuc'][beta // 3][beta % 3] += mom_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = total_energy(E_list)

                # Append new energies.
                pos_e.append(E_tot)

                # Reset the second momentum perturbation.
                self.parameters['P_nuc'][beta // 3][beta % 3] -= mom_pert_strength

            # Perturb the momentum with a negative displacement.
            for beta in range(3*self.natom):
                self.parameters['P_nuc'][beta // 3][beta % 3] -= mom_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = total_energy(E_list)

                # Append new energies.
                neg_e.append(E_tot)

                # Reset the second momentum perturbation.
                self.parameters['P_nuc'][beta // 3][beta % 3] += mom_pert_strength

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                g = (pos_e[beta] - neg_e[beta]) / (2 * mom_pert_strength)
                neg_E.append(g)

            # Reset the first momentum perturbation.
            self.parameters['P_nuc'][alpha // 3][alpha % 3] += mom_pert_strength

        # Compute the momentum Hessian.
        pos_E = np.array(pos_E)
        neg_E = np.array(neg_E)
        pos_E = pos_E.reshape((3 * self.natom, 3 * self.natom))
        neg_E = neg_E.reshape((3 * self.natom, 3 * self.natom))
        momentum_hessian = (pos_E - neg_E) / (2 * mom_pert_strength)

        return momentum_hessian

    def compute_ps_AAT(self, mag_pert_strength, mom_pert_strength, print_level=0):
        # Make sure you have specified a phase-space Hamiltonian.
        assert self.parameters.get('hamiltonian', None) == 'phase-space'

        # Add the nuclear momentum matrix if it isn't already in the dictionary.
        if self.parameters.get('P_nuc', None) == None:
            self.parameters['P_nuc'] = [list(range(3)) for i in range(self.natom)]
            for A in range(self.natom):
                for alpha in range(3):
                    self.parameters['P_nuc'][A][alpha] = 0.0

        # Set properties of the finite difference procedure.
        pos_E = []
        neg_E = []

        # Perturb the momentum in the positive direction.
        for alpha in range(3*self.natom):
            self.parameters['P_nuc'][alpha // 3][alpha % 3] += mom_pert_strength

            pos_e = []
            neg_e = []

            # Perturb the magnetic field with a positive displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] += mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                pos_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] -= mag_pert_strength

            # Perturb the magnetic field with a negative displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] -= mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                neg_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] += mag_pert_strength

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * mag_pert_strength)
                pos_E.append(mu)

            # Reset the momentum perturbation.
            self.parameters['P_nuc'][alpha // 3][alpha % 3] -= mom_pert_strength

        # Perturb the momentum in the negative direction.
        for alpha in range(3*self.natom):
            self.parameters['P_nuc'][alpha // 3][alpha % 3] -= mom_pert_strength

            pos_e = []
            neg_e = []

            # Perturb the magnetic field with a positive displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] += mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                pos_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] -= mag_pert_strength

            # Perturb the magnetic field with a negative displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] -= mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                neg_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] += mag_pert_strength

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * mag_pert_strength)
                neg_E.append(mu)

            # Reset the momentum perturbation.
            self.parameters['P_nuc'][alpha // 3][alpha % 3] += mom_pert_strength

        # Compute the phase-space AAT.
        pos_E = np.array(pos_E)
        neg_E = np.array(neg_E)
        pos_E = pos_E.reshape((3 * self.natom, 3))
        neg_E = neg_E.reshape((3 * self.natom, 3))
        ps_aat = (pos_E - neg_E) / (2 * mom_pert_strength)

        return ps_aat

    def compute_ps_DO_AAT(self, mag_pert_strength, mom_pert_strength, print_level=0):
        # Make sure you have specified a phase-space Hamiltonian.
        assert self.parameters.get('hamiltonian', None) == 'phase-space'

        # Add the nuclear momentum matrix if it isn't already in the dictionary.
        if self.parameters.get('P_nuc', None) == None:
            self.parameters['P_nuc'] = [list(range(3)) for i in range(self.natom)]
            for A in range(self.natom):
                for alpha in range(3):
                    self.parameters['P_nuc'][A][alpha] = 0.0

        # Get equilibrium geometry for distributed origin gauge.
        geom, _, _, _, _ = self.molecule.to_arrays()

        # Set properties of the finite difference procedure.
        pos_E = []
        neg_E = []

        # Perturb the momentum in the positive direction.
        for alpha in range(3*self.natom):
            A = alpha // 3  # atom index

            # Set the gauge origin to atom A for distributed origin.
            self.parameters['gauge_origin'] = list(geom[A])

            self.parameters['P_nuc'][A][alpha % 3] += mom_pert_strength

            pos_e = []
            neg_e = []

            # Perturb the magnetic field with a positive displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] += mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                pos_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] -= mag_pert_strength

            # Perturb the magnetic field with a negative displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] -= mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                neg_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] += mag_pert_strength

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * mag_pert_strength)
                pos_E.append(mu)

            # Reset the momentum perturbation.
            self.parameters['P_nuc'][A][alpha % 3] -= mom_pert_strength

        # Perturb the momentum in the negative direction.
        for alpha in range(3*self.natom):
            A = alpha // 3  # atom index

            # Set the gauge origin to atom A for distributed origin.
            self.parameters['gauge_origin'] = list(geom[A])

            self.parameters['P_nuc'][A][alpha % 3] -= mom_pert_strength

            pos_e = []
            neg_e = []

            # Perturb the magnetic field with a positive displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] += mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                pos_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] -= mag_pert_strength

            # Perturb the magnetic field with a negative displacement.
            for beta in range(3):
                self.parameters['F_mag'][beta] -= mag_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters, print_level)
                E_tot = total_energy(E_list)

                # Append new energies.
                neg_e.append(E_tot)

                # Reset the magnetic field perturbation.
                self.parameters['F_mag'][beta] += mag_pert_strength

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * mag_pert_strength)
                neg_E.append(mu)

            # Reset the momentum perturbation.
            self.parameters['P_nuc'][A][alpha % 3] += mom_pert_strength

        # Remove the gauge origin parameter after computation.
        self.parameters.pop('gauge_origin', None)

        # Compute the phase-space AAT.
        pos_E = np.array(pos_E)
        neg_E = np.array(neg_E)
        pos_E = pos_E.reshape((3 * self.natom, 3))
        neg_E = neg_E.reshape((3 * self.natom, 3))
        ps_aat = (pos_E - neg_E) / (2 * mom_pert_strength)

        return ps_aat
