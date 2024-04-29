"""This script contains a set of functions for finite difference procedures."""

import numpy as np
import psi4
from apyib.hf_wfn import hf_wfn
from apyib.mp2_wfn import mp2_wfn
from apyib.ci_wfn import ci_wfn
from apyib.hamiltonian import Hamiltonian
from apyib.utils import run_psi4
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


    # Computes the energies and wavefunctions for nuclear displacements.
    def nuclear_displacements(self, pert_strength):
        # Set properties of the finite difference procedure.
        pos_e = []
        neg_e = []
        pos_wfns = []
        neg_wfns = []
        pos_basis = []
        neg_basis = []
        pos_t2 = []
        neg_t2 = []

        # Computing energies and wavefunctions with positive displacements.
        print("Computing energies and wavefunctions for positive nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] += pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
           
            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(H.basis_set)).np)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            print("SCF Energy: ", e_tot)

            # Run Psi4.
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                pos_e.append(e_tot)
                pos_wfns.append(pc_C)
                t2 = np.zeros((wfn.ndocc, wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
                pos_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2()
                print("MP2 Energy: ", e_tot + e_MP2)
                
                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_MP2)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID()
                print("CID Energy: ", e_tot + e_CID, "\n")
   
                # Append new energies, wavefunctions, and amplitudes. 
                pos_e.append(e_tot + e_CID)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)
            
            # Store the energies, wavefunction coefficients, and basis set.
            pos_basis.append(H.basis_set)

            # Reset the geometry.
            pert_geom = self.geom

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)
                
            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] -= pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(H.basis_set)).np)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            print("SCF Energy: ", e_tot)

            # Run Psi4.
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                neg_e.append(e_tot)
                neg_wfns.append(pc_C)
                t2 = np.zeros((wfn.ndocc, wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
                neg_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2()
                print("MP2 Energy: ", e_tot + e_MP2)
    
                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_MP2)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID()
                print("CID Energy: ", e_tot + e_CID, "\n")
    
                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_CID)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            neg_basis.append(H.basis_set)

            # Reset the geometry.
            pert_geom = self.geom

        # Reset the molecule's geometry out of loop to ensure further function calls on molecule are not done with a distorted geometry.
        reset_geom = psi4.core.Matrix.from_array(self.geom)
        self.molecule.set_geometry(reset_geom)
        self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        return pos_e, neg_e, pos_wfns, neg_wfns, pos_basis, neg_basis, pos_t2, neg_t2



    # Compute the energies and wavefunctions for electric field perturbations.
    def electric_field_perturbations(self, pert_strength):
        # Set properties of the finite difference procedure.
        pos_e = []
        neg_e = []
        pos_wfns = []
        neg_wfns = []
        pos_basis = []
        neg_basis = []
        pos_t2 = []
        neg_t2 = []

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

            # Run Psi4.
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                pos_e.append(e_tot)
                pos_wfns.append(pc_C)
                t2 = np.zeros((wfn.ndocc, wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
                pos_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2()
                print("MP2 Energy: ", e_tot + e_MP2)
    
                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_MP2)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_CID)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            pos_basis.append(H.basis_set)

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

            # Run Psi4.
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                neg_e.append(e_tot)
                neg_wfns.append(pc_C)
                t2 = np.zeros((wfn.ndocc, wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
                neg_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2()
                print("MP2 Energy: ", e_tot + e_MP2)
    
                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_MP2)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_CID)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            neg_basis.append(H.basis_set)

            # Reset the geometry.
            self.parameters['F_el'][alpha] += pert_strength

        return pos_e, neg_e, pos_wfns, neg_wfns, pos_basis, neg_basis, pos_t2, neg_t2



    # Compute the energies and wavefunctions for magnetic field perturbations.
    def magnetic_field_perturbations(self, pert_strength):
        # Set properties of the finite difference procedure.
        pos_e = []
        neg_e = []
        pos_wfns = []
        neg_wfns = []
        pos_basis = []
        neg_basis = []
        pos_t2 = []
        neg_t2 = []

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
            print("SCF Energy: ", e_tot)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                pos_e.append(e_tot)
                pos_wfns.append(pc_C)
                t2 = np.zeros((wfn.ndocc, wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
                pos_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2()
                print("MP2 Energy: ", e_tot + e_MP2, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_MP2)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_CID)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            pos_basis.append(H.basis_set)

            # Reset the field.
            self.parameters['F_mag'][alpha] -= pert_strength

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative magnetic field perturbations.")
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
            print("SCF Energy: ", e_tot)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                neg_e.append(e_tot)
                neg_wfns.append(pc_C)
                t2 = np.zeros((wfn.ndocc, wfn.ndocc, wfn.nbf-wfn.ndocc, wfn.nbf-wfn.ndocc))
                neg_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2()
                print("MP2 Energy: ", e_tot + e_MP2, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_MP2)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_CID)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            neg_basis.append(H.basis_set)

            # Reset the geometry.
            self.parameters['F_mag'][alpha] += pert_strength

        return pos_e, neg_e, pos_wfns, neg_wfns, pos_basis, neg_basis, pos_t2, neg_t2



    # Computes the energies and wavefunctions for nuclear displacements.
    def nuclear_displacements_SO(self, pert_strength):
        # Set properties of the finite difference procedure.
        pos_e = []
        neg_e = []
        pos_wfns = []
        neg_wfns = []
        pos_basis = []
        neg_basis = []
        pos_t2 = []
        neg_t2 = []

        # Computing energies and wavefunctions with positive displacements.
        print("Computing energies and wavefunctions for positive nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] += pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
    
            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(H.basis_set)).np)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            print("SCF Energy: ", e_tot)

            # Run Psi4.
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                pos_e.append(e_tot)
                pos_wfns.append(pc_C)
                t2 = np.zeros((2*wfn.ndocc, 2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
                pos_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2_SO()
                print("MP2 Energy: ", e_tot + e_MP2)
    
                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_MP2)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID_SO()
                print("CID Energy: ", e_tot + e_CID, "\n")
   
                # Append new energies, wavefunctions, and amplitudes. 
                pos_e.append(e_tot + e_CID)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)
    
            # Store the energies, wavefunction coefficients, and basis set.
            pos_basis.append(H.basis_set)

            # Reset the geometry.
            pert_geom = self.geom

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] -= pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            # Build the Hamiltonian in the AO basis.
            H = Hamiltonian(self.parameters)
            #print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(H.basis_set)).np)

            # Set the Hamiltonian defining this instance of the wavefunction object.
            wfn = hf_wfn(H)

            # Solve the SCF procedure and compute the energy and wavefunction.
            e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
            print("SCF Energy: ", e_tot)

            # Run Psi4.
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                neg_e.append(e_tot)
                neg_wfns.append(pc_C)
                t2 = np.zeros((2*wfn.ndocc, 2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
                neg_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2_SO()
                print("MP2 Energy: ", e_tot + e_MP2)

                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_MP2)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID_SO()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_CID)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            neg_basis.append(H.basis_set)

            # Reset the geometry.
            pert_geom = self.geom

        # Reset the molecule's geometry out of loop to ensure further function calls on molecule are not done with a distorted geometry.
        reset_geom = psi4.core.Matrix.from_array(self.geom)
        self.molecule.set_geometry(reset_geom)
        self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

        return pos_e, neg_e, pos_wfns, neg_wfns, pos_basis, neg_basis, pos_t2, neg_t2



    # Compute the energies and wavefunctions for electric field perturbations.
    def electric_field_perturbations_SO(self, pert_strength):
        # Set properties of the finite difference procedure.
        pos_e = []
        neg_e = []
        pos_wfns = []
        neg_wfns = []
        pos_basis = []
        neg_basis = []
        pos_t2 = []
        neg_t2 = []

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

            # Run Psi4.
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                pos_e.append(e_tot)
                pos_wfns.append(pc_C)
                t2 = np.zeros((2*wfn.ndocc, 2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
                pos_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2_SO()
                print("MP2 Energy: ", e_tot + e_MP2)

                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_MP2)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID_SO()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_CID)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            pos_basis.append(H.basis_set)

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

            # Run Psi4.
            p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
            print("Psi4 Energy: ", p4_rhf_e)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                neg_e.append(e_tot)
                neg_wfns.append(pc_C)
                t2 = np.zeros((2*wfn.ndocc, 2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
                neg_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2_SO()
                print("MP2 Energy: ", e_tot + e_MP2)

                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_MP2)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID_SO()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_CID)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            neg_basis.append(H.basis_set)

            # Reset the geometry.
            self.parameters['F_el'][alpha] += pert_strength

        return pos_e, neg_e, pos_wfns, neg_wfns, pos_basis, neg_basis, pos_t2, neg_t2



    # Compute the energies and wavefunctions for magnetic field perturbations.
    def magnetic_field_perturbations_SO(self, pert_strength):
        # Set properties of the finite difference procedure.
        pos_e = []
        neg_e = []
        pos_wfns = []
        neg_wfns = []
        pos_basis = []
        neg_basis = []
        pos_t2 = []
        neg_t2 = []

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
            print("SCF Energy: ", e_tot)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                pos_e.append(e_tot)
                pos_wfns.append(pc_C)
                t2 = np.zeros((2*wfn.ndocc, 2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
                pos_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2_SO()
                print("MP2 Energy: ", e_tot + e_MP2, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_MP2)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID_SO()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                pos_e.append(e_tot + e_CID)
                pos_wfns.append(pc_C)
                pos_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            pos_basis.append(H.basis_set)

            # Reset the field.
            self.parameters['F_mag'][alpha] -= pert_strength

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative magnetic field perturbations.")
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
            print("SCF Energy: ", e_tot)

            # Computing parameters for the method of choice.
            if self.parameters['method'] == 'RHF':
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)
                neg_e.append(e_tot)
                neg_wfns.append(pc_C)
                t2 = np.zeros((2*wfn.ndocc, 2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc, 2*wfn.nbf-2*wfn.ndocc))
                neg_t2.append(t2)
                print("\n")

            if self.parameters['method'] == 'MP2':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run MP2 code.
                wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_MP2, t2 = wfn_MP2.solve_MP2_SO()
                print("MP2 Energy: ", e_tot + e_MP2, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_MP2)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            if self.parameters['method'] == 'CID':
                # Correct the phase.
                pc_C = compute_phase(wfn.ndocc, wfn.nbf, self.unperturbed_basis, self.unperturbed_wfn, H.basis_set, C)

                # Run CID code.
                wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, pc_C)
                e_CID, t2 = wfn_CID.solve_CID_SO()
                print("CID Energy: ", e_tot + e_CID, "\n")

                # Append new energies, wavefunctions, and amplitudes.
                neg_e.append(e_tot + e_CID)
                neg_wfns.append(pc_C)
                neg_t2.append(t2)

            # Store the energies, wavefunction coefficients, and basis set.
            neg_basis.append(H.basis_set)

            # Reset the geometry.
            self.parameters['F_mag'][alpha] += pert_strength

        return pos_e, neg_e, pos_wfns, neg_wfns, pos_basis, neg_basis, pos_t2, neg_t2



    # Computes the energies and wavefunctions for nuclear displacements.
    def nuclear_and_electric_field_perturbations(self, nuc_pert_strength, elec_pert_strength):
        # Set properties of the finite difference procedure.
        pos_mu = []
        neg_mu = []

        # Computing energies and wavefunctions with positive displacements.
        print("Computing energies and wavefunctions for positive nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] += nuc_pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            pos_e = []
            neg_e = []

            for beta in range(6):
                if beta % 2 == 0:
                    self.parameters['F_el'][beta // 2] += elec_pert_strength
                if beta % 2 == 1:
                    self.parameters['F_el'][beta // 2] -= elec_pert_strength

                # Build the Hamiltonian in the AO basis.
                H = Hamiltonian(self.parameters)

                # Set the Hamiltonian defining this instance of the wavefunction object.
                wfn = hf_wfn(H)

                # Solve the SCF procedure and compute the energy and wavefunction.
                e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
                print("SCF Energy: ", e_tot)

                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
                print("Psi4 Energy: ", p4_rhf_e)

                # Computing parameters for the method of choice.
                if self.parameters['method'] == 'RHF':
                    if beta % 2 == 0:
                        pos_e.append(e_tot)
                    if beta % 2 == 1:
                        neg_e.append(e_tot)
                    print("\n")

                if self.parameters['method'] == 'MP2':
                    # Run MP2 code.
                    wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, C)
                    e_MP2, t2 = wfn_MP2.solve_MP2()
                    print("MP2 Energy: ", e_tot + e_MP2)

                    # Run Psi4.
                    self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                    p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                    print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                    # Append new energies, wavefunctions, and amplitudes.
                    if beta % 2 == 0:
                        pos_e.append(e_tot + e_MP2)
                    if beta % 2 == 1:
                        neg_e.append(e_tot + e_MP2)

                if self.parameters['method'] == 'CID':
                    # Run CID code.
                    wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, C)
                    e_CID, t2 = wfn_CID.solve_CID()
                    print("CID Energy: ", e_tot + e_CID, "\n")

                    # Append new energies, wavefunctions, and amplitudes. 
                    if beta % 2 == 0:
                        pos_e.append(e_tot + e_CID)
                    if beta % 2 == 1:
                        neg_e.append(e_tot + e_CID)

                # Reset the field.
                if beta % 2 == 0:
                    self.parameters['F_el'][beta // 2] -= elec_pert_strength
                if beta % 2 == 1:
                    self.parameters['F_el'][beta // 2] += elec_pert_strength

            # Compute and append electric dipoles.
            for beta in range(3):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * elec_pert_strength)
                pos_mu.append(mu)

            # Reset the geometry.
            pert_geom = self.geom

        # Computing energies and wavefunctions with negative displacements.
        print("Computing energies and wavefunctions for negative nuclear displacements.")
        for alpha in range(3*self.natom):
            pert_geom = np.copy(self.geom)

            # Perturb the geometry.
            pert_geom[alpha // 3][alpha % 3] -= nuc_pert_strength
            pert_geom = psi4.core.Matrix.from_array(pert_geom)
            self.molecule.set_geometry(pert_geom)
            self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()

            pos_e = []
            neg_e = []

            for beta in range(6):
                if beta % 2 == 0:
                    self.parameters['F_el'][beta // 2] += elec_pert_strength
                if beta % 2 == 1:
                    self.parameters['F_el'][beta // 2] -= elec_pert_strength

                # Build the Hamiltonian in the AO basis.
                H = Hamiltonian(self.parameters)

                # Set the Hamiltonian defining this instance of the wavefunction object.
                wfn = hf_wfn(H)

                # Solve the SCF procedure and compute the energy and wavefunction.
                e_elec, e_tot, C = wfn.solve_SCF(self.parameters)
                print("SCF Energy: ", e_tot)

                # Run Psi4.
                self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                p4_rhf_e, p4_rhf_wfn = run_psi4(self.parameters)
                print("Psi4 Energy: ", p4_rhf_e)

                # Computing parameters for the method of choice.
                if self.parameters['method'] == 'RHF':
                    if beta % 2 == 0:
                        pos_e.append(e_tot)
                    if beta % 2 == 1:
                        neg_e.append(e_tot)
                    print("\n")

                if self.parameters['method'] == 'MP2':
                    # Run MP2 code.
                    wfn_MP2 = mp2_wfn(self.parameters, e_elec, e_tot, C)
                    e_MP2, t2 = wfn_MP2.solve_MP2()
                    print("MP2 Energy: ", e_tot + e_MP2)

                    # Run Psi4.
                    self.parameters['geom'] = self.molecule.create_psi4_string_from_molecule()
                    p4_mp2_e, p4_mp2_wfn = run_psi4(self.parameters, 'MP2')
                    print("Psi4 MP2 Energy: ", p4_mp2_e, "\n")

                    # Append new energies, wavefunctions, and amplitudes.
                    if beta % 2 == 0:
                        pos_e.append(e_tot + e_MP2)
                    if beta % 2 == 1:
                        neg_e.append(e_tot + e_MP2)

                if self.parameters['method'] == 'CID':
                    # Run CID code.
                    wfn_CID = ci_wfn(self.parameters, e_elec, e_tot, C)
                    e_CID, t2 = wfn_CID.solve_CID()
                    print("CID Energy: ", e_tot + e_CID, "\n")

                    # Append new energies, wavefunctions, and amplitudes. 
                    if beta % 2 == 0:
                        pos_e.append(e_tot + e_CID)
                    if beta % 2 == 1:
                        neg_e.append(e_tot + e_CID)

                # Reset the field.
                if beta % 2 == 0:
                    self.parameters['F_el'][beta // 2] -= elec_pert_strength
                if beta % 2 == 1:
                    self.parameters['F_el'][beta // 2] += elec_pert_strength

            # Compute and append electric dipoles.
            for beta in range(3):
                mu = -(pos_e[beta] - neg_e[beta]) / (2 * elec_pert_strength)
                neg_mu.append(mu)

            # Reset the geometry.
            pert_geom = self.geom

        return pos_mu, neg_mu







