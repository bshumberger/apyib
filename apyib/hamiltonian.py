"""Contains the Hamiltonian object."""

import psi4
import numpy as np

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

        print("This is the parameters['geom'].")
        print(parameters['geom'])

        # Define the molecule and basis set as properties of the Hamiltonian.
        self.molecule = psi4.geometry(parameters['geom'])

        print("This is the molecule.")
        print(self.molecule.geometry().to_array())

        self.basis_set = psi4.core.BasisSet.build(self.molecule)

        print("This is the geometry.")
        print(psi4.core.Molecule.geometry(psi4.core.BasisSet.molecule(self.basis_set)).np)

        # Use the MintsHelper to get the AO integrals.
        mints = psi4.core.MintsHelper(self.basis_set)

        # Compute AO integrals.
        self.T = mints.ao_kinetic().np       # T_{\mu\nu} = \int \phi_{\mu}^*(r) \left( -\frac{1}{2} \nabla^2_r \right) \phi_{\nu}(r) dr
        self.V = mints.ao_potential().np     # V_{\mu\nu} = \int \phi_{\mu}^*(r) \left( -\sum_A^N \frac{Z}{r_A} \right) \phi_{\nu}(r) dr
        self.ERI = mints.ao_eri().np         # (\mu \nu|\lambda \sigma) = \int \phi_{\mu}^*(r_1) \phi_{\nu}(r_1) r_{12}^{-1} \phi_{\lambda}^*(r_2) \phi_{\sigma}(r_2) dr_1 dr_2
        self.S = mints.ao_overlap().np       # S_{\mu\nu} = \int \phi_{\mu}^*(r) \phi_{\nu}(r) dr

        # Electric dipole AO integrals.
        self.mu_el = mints.ao_dipole()       # \mu^{el}_{\mu\nu, \alpha} = -e \int \phi_{\mu}^*(r) r_{\alpha} \phi_{\nu}(r) dr
        for alpha in range(3):
            self.mu_el[alpha] = self.mu_el[alpha].np
            self.mu_el[alpha] = self.mu_el[alpha].astype('complex128')

        # Magnetic dipole AO integrals.
        self.mu_mag = mints.ao_angular_momentum()    # \mu^{mag}_{\mu\nu, \alpha} = - \frac(e}{2 m_e} \int \phi_{\mu}^*(r) (r x p)_{\alpha} \phi_{\nu}(r) dr
        for alpha in range(3):
            self.mu_mag[alpha] = -0.5j * self.mu_mag[alpha].np
            self.mu_mag[alpha] = self.mu_mag[alpha].astype('complex128')

        # Compute the nuclear repulsion energy.
        F_el = [0.0, 0.0, 0.0]
        for alpha in range(3):
            F_el[alpha] += parameters['F_el'][alpha] * -1
        self.E_nuc = self.molecule.nuclear_repulsion_energy(F_el)

        # Add electric and magnetic potentials to the core Hamiltonian.
        self.V = self.V.astype('complex128')
        for alpha in range(3):
            self.V -=  parameters['F_el'][alpha] * self.mu_el[alpha] + parameters['F_mag'][alpha] * self.mu_mag[alpha]

