import numpy as np
import apyib
import pytest


def _random_complex_inputs(no=6, nv=8, seed=90210):
    """Complex integrals and amplitudes carrying only the symmetries that survive
    complex orbitals.

    <pq||rs> is antisymmetric within the bra pair and within the ket pair, but the
    bra-ket exchange <pq||rs> = <rs||pq> holds for real orbitals only and is
    deliberately not imposed. The Fock matrix is likewise left non-symmetric, so
    f_pq and f_qp are distinguishable.
    """
    rng = np.random.default_rng(seed)

    def crand(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def antisymmetrized(*shape):
        Z = crand(*shape)
        return Z - Z.swapaxes(0,1) - Z.swapaxes(2,3) + Z.swapaxes(0,1).swapaxes(2,3)

    nt = no + nv
    o_ = slice(0, no)
    v_ = slice(no, nt)

    F = crand(nt, nt)
    G = antisymmetrized(nt, nt, nt, nt)
    t1, t2 = crand(no, nv), antisymmetrized(no, no, nv, nv)
    l1, l2 = crand(no, nv), antisymmetrized(no, no, nv, nv)

    assert not np.allclose(F, F.T)
    assert not np.allclose(G, G.swapaxes(0,2).swapaxes(1,3))

    return F, G, t1, t2, l1, l2, o_, v_


def test_lambda_residual_drivers_are_the_bare_integrals():
    """With all amplitudes zero the Lambda residuals collapse to their drivers.

    Every term of the Lambda equations except the driver carries at least one t or
    one lambda amplitude, so setting them all to zero must leave exactly f_kc and
    <kl||cd>.

    These drivers are the bra-ket transposes of the T drivers, f_ai and <ab||ij>.
    The two orderings are the same numbers for real orbitals, so writing a Lambda
    driver with the T orientation would pass every real-arithmetic test and be
    wrong the moment a magnetic field is switched on. Hence the complex inputs.
    """
    F, G, t1, t2, l1, l2, o_, v_ = _random_complex_inputs()
    cc = apyib.cc_wfn.cc_wfn
    zero1, zero2 = np.zeros_like(t1), np.zeros_like(t2)

    r_L1 = cc._lambda_singles_residual_SO(None, F, G, zero1, zero2, zero1, zero2, o_, v_)
    r_L2 = cc._lambda_doubles_residual_SO(None, F, G, zero1, zero2, zero1, zero2, o_, v_)

    assert np.allclose(r_L1, F[o_,v_], rtol=0, atol=1e-13)
    assert np.allclose(r_L2, G[o_,o_,v_,v_], rtol=0, atol=1e-13)

    # The T orientation must not reproduce them.
    assert not np.allclose(r_L1, F.swapaxes(0,1)[o_,v_])
    assert not np.allclose(r_L2, G.swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_])


def test_lambda_doubles_residual_antisymmetry():
    """The Lambda doubles residual projects onto an antisymmetrized determinant.

    Exchanging the two occupied labels, or the two virtual labels, must flip its
    sign. A wrong transpose axis or sign in the permutation grouping breaks this
    while leaving a plausible-looking residual.
    """
    F, G, t1, t2, l1, l2, o_, v_ = _random_complex_inputs()
    r_L2 = apyib.cc_wfn.cc_wfn._lambda_doubles_residual_SO(
        None, F, G, t1, t2, l1, l2, o_, v_)

    scale = np.max(np.abs(r_L2))
    assert scale > 1.0

    assert np.max(np.abs(r_L2 + r_L2.swapaxes(0,1))) < 1e-10 * scale
    assert np.max(np.abs(r_L2 + r_L2.swapaxes(2,3))) < 1e-10 * scale
    assert np.max(np.abs(r_L2 - r_L2.swapaxes(0,1).swapaxes(2,3))) < 1e-10 * scale


def test_lambda_residuals_are_affine_in_lambda():
    """The Lambda equations are linear in the Lambda amplitudes.

    Each term carries at most one lambda factor, so the residual is affine: with
    the t amplitudes held fixed, r(s * lambda) - r(0) must scale exactly with s.
    A term carrying a stray second lambda, or a lambda where a t belongs, breaks
    this and nothing else would catch it.
    """
    F, G, t1, t2, l1, l2, o_, v_ = _random_complex_inputs()
    cc = apyib.cc_wfn.cc_wfn
    zero1, zero2 = np.zeros_like(l1), np.zeros_like(l2)
    s = 3.7

    for residual in (cc._lambda_singles_residual_SO, cc._lambda_doubles_residual_SO):
        r_zero = residual(None, F, G, t1, t2, zero1, zero2, o_, v_)
        r_one = residual(None, F, G, t1, t2, l1, l2, o_, v_)
        r_scaled = residual(None, F, G, t1, t2, s * l1, s * l2, o_, v_)

        scale = np.max(np.abs(r_scaled))
        assert np.max(np.abs((r_scaled - r_zero) - s * (r_one - r_zero))) < 1e-10 * scale


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


def _parameters(**overrides):
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CCSD_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 200}
    parameters.update(overrides)
    return parameters


def _spin_orbital_integrals(parameters, wfn, wfn_CC):
    """The same F and G the solver builds internally."""
    F = apyib.utils.compute_F_SO(wfn, wfn_CC.F_MO)
    ERI = apyib.utils.compute_ERI_SO(wfn, wfn_CC.ERI_MO).swapaxes(1, 2)
    return F, ERI - ERI.swapaxes(2, 3)


@pytest.mark.parametrize("freeze_core", [False, True])
def test_converged_lambda_annihilates_its_residuals(freeze_core):
    """The Lambda equations are solved when their residuals vanish.

    This is the defining condition, 0 = dL/dt, so it is checked directly on the
    returned amplitudes rather than inferred from the iteration having stopped.
    The residuals are compared against the energy denominators they are divided
    by, since that combination is the amplitude increment the solver drives to
    zero.
    """
    parameters = _parameters(freeze_core=freeze_core, solve_lambda=True)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)
    wfn.solve_SCF(parameters)
    wfn_CC = apyib.cc_wfn.cc_wfn(parameters, wfn)
    E_CCSD, t1, t2, l1, l2 = wfn_CC.solve_CCSD_SO(solving_lambda=True)

    F, G = _spin_orbital_integrals(parameters, wfn, wfn_CC)
    o_, v_ = wfn_CC.I_list[1], wfn_CC.I_list[2]

    eps_o = np.repeat(wfn_CC.eps_o, 2)
    eps_v = np.repeat(wfn_CC.eps_v, 2)
    D_ia = eps_o.reshape(-1, 1) - eps_v
    D_ijab = (eps_o.reshape(-1, 1, 1, 1) + eps_o.reshape(-1, 1, 1)
              - eps_v.reshape(-1, 1) - eps_v)

    cc = apyib.cc_wfn.cc_wfn
    r_L1 = cc._lambda_singles_residual_SO(None, F, G, t1, t2, l1, l2, o_, v_)
    r_L2 = cc._lambda_doubles_residual_SO(None, F, G, t1, t2, l1, l2, o_, v_)

    assert np.max(np.abs(r_L1 / D_ia)) < 1e-11
    assert np.max(np.abs(r_L2 / D_ijab)) < 1e-11

    # The doubles manifold antisymmetry must survive the solve.
    assert np.allclose(l2, -l2.swapaxes(0, 1))
    assert np.allclose(l2, -l2.swapaxes(2, 3))

    # Lambda and t agree only at lowest order; identical amplitudes would mean the
    # iteration never moved off its starting guess.
    assert np.max(np.abs(l2 - t2)) > 1e-6


def test_solve_lambda_does_not_disturb_the_t_amplitudes():
    """Requesting Lambda must change nothing about the CCSD solution itself.

    The Lambda equations take the converged t amplitudes as fixed input, so the
    energy and both t amplitudes have to come back bit-for-bit identical whether
    or not Lambda was requested.
    """
    E_off, T_off, _, _ = apyib.energy.energy(_parameters())
    E_on, T_on, _, _ = apyib.energy.energy(_parameters(solve_lambda=True))

    assert E_off[1] == E_on[1]
    assert np.array_equal(T_off[1], T_on[1])
    assert np.array_equal(T_off[2], T_on[2])

    # Without the flag the Lambda slots stay at their sentinel value.
    assert T_off[3] == 0 and T_off[4] == 0

    # With it they carry amplitudes shaped like their t counterparts.
    assert T_on[3].shape == T_on[1].shape
    assert T_on[4].shape == T_on[2].shape


def test_lambda_converges_without_diis():
    """The bare fixed-point iteration must converge on its own.

    DIIS extrapolates to the point where the error vector vanishes, which is the
    solution regardless of the direction or scale of the step it was handed. A
    wrong step would therefore still reach the right answer with DIIS on, while
    diverging without it. Running the iteration undamped is what pins the update
    itself, and both routes must land on the same amplitudes.
    """
    l_diis = apyib.energy.energy(_parameters(solve_lambda=True, DIIS=True))[1]
    l_bare = apyib.energy.energy(_parameters(solve_lambda=True, DIIS=False))[1]

    assert np.max(np.abs(l_bare[3] - l_diis[3])) < 1e-10
    assert np.max(np.abs(l_bare[4] - l_diis[4])) < 1e-10
