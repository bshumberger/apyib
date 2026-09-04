import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

# H2O2 geometry used by the analytic-derivative test suite (test_015 onward).
H2O2_manuscript = """
H -1.780954530308296   1.411647335546379   0.872055376436941
H  1.780954530308296  -1.411647335546379   0.872055376436941
O -1.371214332646589  -0.115525249760340  -0.054947416764017
O  1.371214332646589   0.115525249760340  -0.054947416764017
no_com
no_reorient
symmetry c1
units bohr
"""

def test_ccsd_so_h2o2_STO_3G():
    # Set parameters for the calculation.
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CCSD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CCSD reference value.
    psi4_CCSD = -148.86781297996708

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CCSD')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CCSD Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CCSD Code and Reference: ", apyib_E_tot - psi4_CCSD)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_CCSD) < 1e-11)

def test_ccsd_so_h2o2_STO_3G_fc():
    # Set parameters for the calculation.
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CCSD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CCSD reference value.
    psi4_CCSD = -148.86768203047497

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CCSD')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CCSD Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CCSD Code and Reference: ", apyib_E_tot - psi4_CCSD)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_CCSD) < 1e-11)

def test_ccsd_so_h2o_6_31Gd():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G*',
                  'method': 'CCSD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CCSD reference value.
    psi4_CCSD = -76.18481584171214

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CCSD')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CCSD Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CCSD Code and Reference: ", apyib_E_tot - psi4_CCSD)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_CCSD) < 1e-11)

def test_ccsd_so_h2o_6_31Gd_fc():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O"],
                  'basis': '6-31G*',
                  'method': 'CCSD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting CCSD reference value.
    psi4_CCSD = -76.18272492114800

    # Run Psi4.
    p4_E_tot, p4_wfn = apyib.utils.run_psi4(parameters, 'CCSD')
    psi4.core.clean()

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    apyib_E_tot = E_list[0] + E_list[1] + E_list[2]

    # Print energies and energy difference between apyib code and Psi4.
    print("Electronic Hartree-Fock Energy: ", E_list[0])
    print("Electronic CCSD Energy: ", E_list[1])
    print("Total Energy: ", apyib_E_tot)
    print("Energy Difference between Homemade CCSD Code and Reference: ", apyib_E_tot - psi4_CCSD)

    assert(abs(apyib_E_tot - p4_E_tot) < 1e-11)
    assert(abs(apyib_E_tot - psi4_CCSD) < 1e-11)

def test_ccsd_so_direct_solver_matches_energy_pipeline():
    """The cc_wfn solver and the energy() dispatch must agree.

    This exercises the registration wiring (config._VALID_METHODS,
    utils.get_slices, and energy._run_post_hf) independently of the residual
    equations themselves.
    """
    # Set parameters for the calculation.
    parameters = {'geom': moldict["(H2)_2"],
                  'basis': 'STO-6G',
                  'method': 'CCSD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Run the solver directly.
    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)
    apyib_E, apyib_wfn = wfn.solve_SCF(parameters)
    wfn_CC = apyib.cc_wfn.cc_wfn(parameters, wfn)
    direct_E_CCSD, t1, t2, l1, l2 = wfn_CC.solve_CCSD_SO()

    # Run the same calculation through the energy() dispatch.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)

    print("Direct CCSD correlation energy: ", direct_E_CCSD)
    print("Pipeline CCSD correlation energy: ", E_list[1])

    assert(abs(direct_E_CCSD - E_list[1]) < 1e-12)
    assert(np.max(np.abs(t1 - T_list[1])) < 1e-12)
    assert(np.max(np.abs(t2 - T_list[2])) < 1e-12)

def test_ccsd_so_doubles_residual_antisymmetry_complex():
    """The doubles residual must be antisymmetric in i,j and in a,b.

    < Phi_ij^ab | H_bar | Phi_0 > projects onto an antisymmetrized determinant,
    so exchanging the two occupied labels, or the two virtual labels, must flip
    the sign of the residual. The folded implementation obtains the sixty-three
    explicit terms by applying P(i,j) and P(a,b) to thirty-one representatives,
    and a wrong transpose axis or a wrong sign in that folding breaks this
    identity while still converging to a plausible energy.

    The check runs in complex arithmetic on purpose. The residual references
    <pq||rs> and f_pq in orderings that are only interchangeable for real
    orbitals, so a bra-ket inversion is invisible on the real axis and would
    surface first as a wrong finite-difference AAT once F_mag is switched on.
    """
    rng = np.random.default_rng(20260904)

    def crand(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    # Spin orbitals: a small occupied/virtual split is enough to fix the identity.
    no, nv = 6, 8
    nt = no + nv
    o_ = slice(0, no)
    v_ = slice(no, nt)

    # Fock matrix, deliberately not symmetric so f_pq and f_qp are distinguishable.
    F = crand(nt, nt)

    # Antisymmetrized integrals: <pq||rs> = -<pq||sr> = -<qp||rs>. Bra-ket pair
    # exchange, <pq||rs> = <rs||pq>, is deliberately NOT imposed; it holds only
    # for real orbitals, and imposing it here would hide an inverted term.
    X = crand(nt, nt, nt, nt)
    G = X - X.swapaxes(0,1) - X.swapaxes(2,3) + X.swapaxes(0,1).swapaxes(2,3)
    assert not np.allclose(G, G.swapaxes(0,2).swapaxes(1,3))

    # Amplitudes carrying the physical antisymmetry of the doubles manifold.
    t1 = crand(no, nv)
    Y = crand(no, no, nv, nv)
    t2 = Y - Y.swapaxes(0,1) - Y.swapaxes(2,3) + Y.swapaxes(0,1).swapaxes(2,3)

    # The residual and its intermediates read nothing off the wavefunction object,
    # so they are evaluated without building one.
    cc = apyib.cc_wfn.cc_wfn
    tau, tau_tilde = cc._tau_s_SO(None, t1, t2)
    F_ae = cc._F_ae_SO(None, F, G, t1, tau_tilde, o_, v_)
    F_mi = cc._F_mi_SO(None, F, G, t1, tau_tilde, o_, v_)
    F_me = cc._F_me_SO(None, F, G, t1, o_, v_)
    W_mnij = cc._W_mnij_SO(None, G, t1, tau, o_, v_)
    W_abef = cc._W_abef_SO(None, G, t1, o_, v_)
    W_mbej = cc._W_mbej_SO(None, G, t1, t2, o_, v_)
    r_T2 = cc._doubles_residual_SO(None, G, t1, t2, tau, F_ae, F_mi, F_me,
                                     W_mnij, W_abef, W_mbej, o_, v_)

    scale = np.max(np.abs(r_T2))
    assert scale > 1.0

    assert np.max(np.abs(r_T2 + r_T2.swapaxes(0,1))) < 1e-10 * scale
    assert np.max(np.abs(r_T2 - r_T2.swapaxes(0,1).swapaxes(2,3))) < 1e-10 * scale
    assert np.max(np.abs(r_T2 + r_T2.swapaxes(2,3))) < 1e-10 * scale
