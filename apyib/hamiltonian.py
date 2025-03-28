"""Contains the Hamiltonian object."""

import psi4
import numpy as np
import opt_einsum as oe

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

