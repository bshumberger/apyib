"""This script contains a set of streamlined functions for finite difference procedures."""

import numpy as np
import psi4
from apyib.energy import energy
from apyib.energy import phase_corrected_energy
#from apyib.rotational_strength import compute_mo_overlap
#from apyib.rotational_strength import compute_phase


class finite_difference(object):
    """
    Finite difference object.
    """
    # Defines the properties of the finite difference procedure.
    def __init__(self, parameters, unperturbed_basis, unperturbed_C):
        self.parameters = parameters
        self.molecule = psi4.geometry(self.parameters['geom'])
        self.geom = self.molecule.geometry().np
        self.natom = self.molecule.natom()
        self.unperturbed_basis = unperturbed_basis
        self.unperturbed_C = unperturbed_C



    def compute_Hessian(self, nuc_pert_strength):
        # Set properties of the finite difference procedure.
        pos_E = []
        neg_E = []

        # Computing energies and wavefunctions with positive displacements.
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] += nuc_pert_strength
            pert_geom_alpha = np.copy(pert_geom)

            pos_e = []
            neg_e = []

            # Perturb the geometry with another positive displacement.
            for beta in range(3*self.natom):
                pert_geom_alpha[beta // 3][beta % 3] += nuc_pert_strength
                pert_geom_alpha_beta = psi4.core.Matrix.from_array(pert_geom_alpha)
                self.molecule.set_geometry(pert_geom_alpha_beta)
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append new energies. 
                pos_e.append(E_tot)

                # Reset the second geometric perturbation.
                pert_geom_alpha = np.copy(pert_geom)

            # Perturb the geometry with a negative displacement.
            for beta in range(3*self.natom):
                pert_geom_alpha[beta // 3][beta % 3] -= nuc_pert_strength
                pert_geom_alpha_beta = psi4.core.Matrix.from_array(pert_geom_alpha)
                self.molecule.set_geometry(pert_geom_alpha_beta)
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append new energies. 
                neg_e.append(E_tot)

                # Reset the second geometric perturbation.
                pert_geom_alpha = np.copy(pert_geom)

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                g = (pos_e[beta] - neg_e[beta]) / (2 * nuc_pert_strength)
                pos_E.append(g)

            # Reset the geometry.
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Computing energies and wavefunctions with negative displacements.
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] -= nuc_pert_strength
            pert_geom_alpha = np.copy(pert_geom)

            pos_e = []
            neg_e = []

            # Perturb the geometry with another positive displacement.
            for beta in range(3*self.natom):
                pert_geom_alpha[beta // 3][beta % 3] += nuc_pert_strength
                pert_geom_alpha_beta = psi4.core.Matrix.from_array(pert_geom_alpha)
                self.molecule.set_geometry(pert_geom_alpha_beta)
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append new energies. 
                pos_e.append(E_tot)

                # Reset the second geometric perturbation.
                pert_geom_alpha = np.copy(pert_geom)

            # Perturb the geometry with a negative displacement.
            for beta in range(3*self.natom):
                pert_geom_alpha[beta // 3][beta % 3] -= nuc_pert_strength
                pert_geom_alpha_beta = psi4.core.Matrix.from_array(pert_geom_alpha)
                self.molecule.set_geometry(pert_geom_alpha_beta)
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append new energies. 
                neg_e.append(E_tot)

                # Reset the second geometric perturbation.
                pert_geom_alpha = np.copy(pert_geom)

            # Compute and append gradients.
            for beta in range(len(pos_e)):
                g = (pos_e[beta] - neg_e[beta]) / (2 * nuc_pert_strength)
                neg_E.append(g)

            # Reset the geometry.
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Compute the Hessian [E_h / a_0**2].
        pos_E = np.array(pos_E)
        neg_E = np.array(neg_E)
        pos_E = pos_E.reshape((3 * self.natom, 3 * self.natom))
        neg_E = neg_E.reshape((3 * self.natom, 3 * self.natom))
        hessian = (pos_E - neg_E) / (2 * nuc_pert_strength)

        return hessian



    def compute_APT(self, nuc_pert_strength, elec_pert_strength):
        # Set properties of the finite difference procedure.
        pos_mu = [] 
        neg_mu = [] 

        # Computing energies and wavefunctions with positive nuclear displacements.
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] += nuc_pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            pos_e = [] 
            neg_e = [] 

            # Perturb the electric field in the positive direction.
            for beta in range(3):
                self.parameters['F_el'][beta] += elec_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append the energy based on the perturbation.
                pos_e.append(E_tot)

                # Reset the field.
                self.parameters['F_el'][beta] -= elec_pert_strength

            # Perturb the electric field in the negative direction.
            for beta in range(3):
                self.parameters['F_el'][beta] -= elec_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append the energy based on the perturbation.
                neg_e.append(E_tot)

                # Reset the field.
                self.parameters['F_el'][beta] += elec_pert_strength

            # Compute and append electric dipoles.
            for beta in range(3):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * elec_pert_strength)
                pos_mu.append(mu)

            # Reset the geometry.
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Computing energies and wavefunctions with negative nuclear displacements.
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] -= nuc_pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            pos_e = []
            neg_e = []

            # Perturb the electric field in the positive direction.
            for beta in range(3):
                self.parameters['F_el'][beta] += elec_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append the energy based on the perturbation.
                pos_e.append(E_tot)

                # Reset the field.
                self.parameters['F_el'][beta] -= elec_pert_strength

            # Perturb the electric field in the negative direction.
            for beta in range(3):
                self.parameters['F_el'][beta] -= elec_pert_strength

                # Compute energy.
                E_list, T_list, C, basis = energy(self.parameters)
                E_tot = E_list[0] + E_list[1] + E_list[2]

                # Append the energy based on the perturbation.
                neg_e.append(E_tot)

                # Reset the field.
                self.parameters['F_el'][beta] += elec_pert_strength

            # Compute and append electric dipoles.
            for beta in range(3):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * elec_pert_strength)
                neg_mu.append(mu)

            # Reset the geometry.
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Compute the APTs.
        pos_mu = np.array(pos_mu)
        neg_mu = np.array(neg_mu)
        pos_mu = pos_mu.reshape((3 * self.natom, 3)) 
        neg_mu = neg_mu.reshape((3 * self.natom, 3)) 
        P = (pos_mu - neg_mu) / (2 * nuc_pert_strength)

        return P



    def compute_AAT(self, nuc_pert_strength, mag_pert_strength):
        # Set properties of the nuclear finite difference procedure.
        nuc_pos_C = []
        nuc_neg_C = []
        nuc_pos_basis = []
        nuc_neg_basis = []
        nuc_pos_T = []
        nuc_neg_T = []

        # Set properties of the magnetic field finite difference procedure.
        mag_pos_C = []
        mag_neg_C = []
        mag_pos_basis = []
        mag_neg_basis = []
        mag_pos_T = []
        mag_neg_T = []

        # Computing energies and wavefunctions with positive displacements.
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] += nuc_pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            # Compute energy.
            E_list, T_list, C, basis = phase_corrected_energy(self.parameters, self.unperturbed_basis, self.unperturbed_C)
            E_tot = E_list[0] + E_list[1] + E_list[2]
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(basis)).np)
            #print(E_tot)

            # Append new wavefunction coefficients, amplitudes, and basis set. 
            nuc_pos_C.append(C)
            nuc_pos_T.append(T_list)
            nuc_pos_basis.append(basis)

            # Reset the geometry.
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Computing energies and wavefunctions with negative displacements.
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] -= nuc_pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            # Compute energy.
            E_list, T_list, C, basis = phase_corrected_energy(self.parameters, self.unperturbed_basis, self.unperturbed_C)
            E_tot = E_list[0] + E_list[1] + E_list[2]
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(basis)).np)
            #print(E_tot)

            # Append new wavefunction coefficients, amplitudes, and basis set. 
            nuc_neg_C.append(C)
            nuc_neg_T.append(T_list)
            nuc_neg_basis.append(basis)

            # Reset the geometry.
            self.molecule.set_geometry(psi4.core.Matrix.from_array(self.geom))
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        # Perturb the magnetic field in the positive direction.
        for beta in range(3):
            self.parameters['F_mag'][beta] += mag_pert_strength

            # Compute energy.
            E_list, T_list, C, basis = phase_corrected_energy(self.parameters, self.unperturbed_basis, self.unperturbed_C)
            E_tot = E_list[0] + E_list[1] + E_list[2]
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(basis)).np)
            #print(self.parameters['F_mag'])
            #print(E_tot)

            # Append new wavefunction coefficients, amplitudes, and basis set. 
            mag_pos_C.append(C)
            mag_pos_T.append(T_list)
            mag_pos_basis.append(basis)

            # Reset the field.
            self.parameters['F_mag'][beta] -= mag_pert_strength

        # Perturb the magnetic field in the negative direction.
        for beta in range(3):
            self.parameters['F_mag'][beta] -= mag_pert_strength

            # Compute energy.
            E_list, T_list, C, basis = phase_corrected_energy(self.parameters, self.unperturbed_basis, self.unperturbed_C)
            E_tot = E_list[0] + E_list[1] + E_list[2]
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(basis)).np)
            #print(self.parameters['F_mag'])
            #print(E_tot)

            # Append new wavefunction coefficients, amplitudes, and basis set. 
            mag_neg_C.append(C)
            mag_neg_T.append(T_list)
            mag_neg_basis.append(basis)

            # Reset the field.
            self.parameters['F_mag'][beta] += mag_pert_strength

        return nuc_pos_C, nuc_neg_C, nuc_pos_basis, nuc_neg_basis, nuc_pos_T, nuc_neg_T, mag_pos_C, mag_neg_C, mag_pos_basis, mag_neg_basis, mag_pos_T, mag_neg_T












