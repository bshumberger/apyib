"""This script contains a set of functions for finite difference procedures."""

import numpy as np
import psi4
from apyib.hf_wfn import hf_wfn
from apyib.hamiltonian import Hamiltonian
from apyib.utils import run_psi4



class finite_difference(object):
    """
    Finite difference object.
    """
    # Defines the properties of the finite difference procedure.
    def __init__(self, parameters):
        self.parameters = parameters
        self.molecule = psi4.geometry(parameters['geom'])
        self.geom = self.molecule.geometry().np
        self.natom = self.molecule.natom()
        self.pos_e = []
        self.neg_e = []
        self.pos_wfns = []
        self.neg_wfns = []
        self.pos_basis = []
        self.neg_basis = []



    # Computes the energies and wavefunctions for nuclear displacements.
    def nuclear_displacements(self, pert_strength):
        # Computing energies and wavefunctions with positive displacements.
        print("Computing energies and wavefunctions for positive nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)
            
            # Perturb the geometry.
            pert_geom[alpha // 3, alpha % 3] += pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            
            #print(self.parameters['geom'])

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            #print("SCF Energy: ", e_tot)

            # Run Psi4.
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            #print("Psi4 Energy: ", p4_rhf_e, "\n")

            # Store the energies, wavefunction coefficients, and basis set.
            self.pos_e.append(e_tot)
            self.pos_wfns.append(C)
            self.pos_basis.append(H.basis_set)

            # Reset the geometry.
            pert_geom = self.geom

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)
                
            # Perturb the geometry.
            pert_geom[alpha // 3, alpha % 3] -= pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            #print(self.parameters['geom'])

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            #print("SCF Energy: ", e_tot)

            # Run Psi4.
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            #print("Psi4 Energy: ", p4_rhf_e, "\n")

            # Store the energies, wavefunction coefficients, and basis set.
            self.neg_e.append(e_tot)
            self.neg_wfns.append(C)
            self.neg_basis.append(H.basis_set)

            # Reset the geometry.
            pert_geom = self.geom

        # Reset the molecule's geometry out of loop to ensure further function calls on molecule are not done with a distorted geometry.
        reset_geom = psi4.core.Matrix.from_array(self.geom)
        self.molecule.set_geometry(reset_geom)
        self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        return self.pos_e, self.neg_e, self.pos_wfns, self.neg_wfns, self.pos_basis, self.neg_basis



    # Compute the energies and wavefunctions for electric field perturbations.
    def electric_field_perturbations(self, pert_strength):
        # Computing energies and wavefunctions with positive displacements.
        print("Computing energies and wavefunctions for positive electric field perturbations.")
        for alpha in range(3):
            self.parameters['F_el'][alpha] += pert_strength
            #print(self.parameters['F_el'])

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            print("SCF Energy: ", e_tot)

            # Store the energies, wavefunction coefficients, and basis set.
            self.pos_e.append(e_tot)
            self.pos_wfns.append(C)
            self.pos_basis.append(H.basis_set)

            # Reset the field.
            self.parameters['F_el'][alpha] -= pert_strength

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative electric field perturbations.")
        for alpha in range(3):
            self.parameters['F_el'][alpha] -= pert_strength
            #print(self.parameters['F_el'])

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            print("SCF Energy: ", e_tot)

            # Store the energies, wavefunction coefficients, and basis set.
            self.neg_e.append(e_tot)
            self.neg_wfns.append(C)
            self.neg_basis.append(H.basis_set)

            # Reset the geometry.
            self.parameters['F_el'][alpha] += pert_strength

        return self.pos_e, self.neg_e, self.pos_wfns, self.neg_wfns, self.pos_basis, self.neg_basis



    # Compute the energies and wavefunctions for magnetic field perturbations.
    def magnetic_field_perturbations(self, pert_strength):
        # Computing energies and wavefunctions with positive displacements.
        print("Computing energies and wavefunctions for positive magnetic field perturbations.")
        for alpha in range(3):
            self.parameters['F_mag'][alpha] += pert_strength
            #print(self.parameters['F_mag'])
            #print(self.parameters['geom'])

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            #print("SCF Energy: ", e_tot)

            # Store the energies, wavefunction coefficients, and basis set.
            self.pos_e.append(e_tot)
            self.pos_wfns.append(C)
            self.pos_basis.append(H.basis_set)

            # Reset the field.
            self.parameters['F_mag'][alpha] -= pert_strength

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative electric field perturbations.")
        for alpha in range(3):
            self.parameters['F_mag'][alpha] -= pert_strength
            #print(self.parameters['F_mag'])
            #print(self.parameters['geom'])

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            #print("SCF Energy: ", e_tot)

            # Store the energies, wavefunction coefficients, and basis set.
            self.neg_e.append(e_tot)
            self.neg_wfns.append(C)
            self.neg_basis.append(H.basis_set)

            # Reset the geometry.
            self.parameters['F_mag'][alpha] += pert_strength

        return self.pos_e, self.neg_e, self.pos_wfns, self.neg_wfns, self.pos_basis, self.neg_basis



