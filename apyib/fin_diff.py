"""This script contains a set of streamlined functions for finite difference procedures."""

import numpy as np
import psi4
from apyib.energy import energy
from apyib.rotational_strength import compute_mo_overlap
from apyib.rotational_strength import compute_phase


class finite_difference(object):
    """
    Finite difference object.
    """
    # Defines the properties of the finite difference procedure.
    def __init__(self, parameters, unperturbed_basis, unperturbed_wfn):
        self.parameters = parameters
        self.molecule = psi4.geometry(parameters['geom'])
        self.geom = self.molecule.geometry().np
        self.natom = self.molecule.natom()
        self.unperturbed_basis = unperturbed_basis
        self.unperturbed_wfn = unperturbed_wfn

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
            pert_geom = np.copy(self.geom)

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
            pert_geom = np.copy(self.geom)

        # Compute the Hessian [E_h / a_0**2].
        pos_E = np.array(pos_E)
        neg_E = np.array(neg_E)
        pos_E = pos_E.reshape((3 * self.natom, 3 * self.natom))
        neg_E = neg_E.reshape((3 * self.natom, 3 * self.natom))
        hessian = (pos_E - neg_E) / (2 * nuc_pert_strength)

        return hessian
