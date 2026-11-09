# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`apyib` is a Python library for quantum chemical calculations, focused on computing Vibrational Circular Dichroism (VCD) spectra. It implements RHF, MP2, CID, and CISD methods with both finite-difference and analytic derivative approaches for molecular properties (Hessian, APTs, AATs), plus a spin-orbital CCSD energy solver (`cc_wfn.py`) that is being built out toward CCSD-level AATs.

**Core dependency:** [Psi4](https://psicode.org/) provides AO integrals, basis set handling, and geometry management. All calculations require a working Psi4 installation. Other key dependencies: `numpy`, `opt_einsum`, `scipy`.

## Commands

```bash
# Development install (the test extra pulls in pytest-xdist, required by -n auto below)
pip install -e ".[test]"

# Run fast tests (excludes slow-marked VCD/analytic-Hessian tests)
pytest apyib/tests/ -m "not slow"

# Run fast tests in parallel (recommended; set OMP_NUM_THREADS=1 to avoid BLAS contention).
# -n auto needs pytest-xdist; if it errors with "unrecognized arguments: -n auto",
# the env predates the test extra -- reinstall with pip install -e ".[test]", or drop -n auto.
OMP_NUM_THREADS=1 pytest apyib/tests/ -m "not slow" -n auto

# Run slow tests explicitly (H2O2/aug-cc-pVDZ analytic suite; ~25 min)
pytest apyib/tests/ -m slow

# Run a single test file
pytest apyib/tests/test_002_RHF.py

# Run a specific test function
pytest apyib/tests/test_002_RHF.py::test_rhf_h2o_sto_3g -v
```

## Architecture

All calculations are driven by a `parameters` dictionary:

```python
parameters = {
    'geom': '<psi4-format geometry string>',
    'basis': 'STO-3G',
    'method': 'RHF',        # RHF, MP2, CID, CISD, MP2_SO, CID_SO, CISD_SO, CCSD_SO
    'e_convergence': 1e-12,
    'd_convergence': 1e-12,
    'DIIS': True,
    'freeze_core': False,
    'F_el': [0.0, 0.0, 0.0],   # Electric field perturbation
    'F_mag': [0.0, 0.0, 0.0],  # Magnetic field perturbation
    'F_mom': [0.0, 0.0, 0.0],  # Momentum (nabla) perturbation, for finite-difference VG APTs
    'max_iterations': 120,
    # Optional:
    'isotopes': {atom_idx: mass},  # Isotopic substitutions
    'P_nuc': ...,                  # Nuclear momentum (phase-space)
}
```

### Module Responsibilities

| Module | Class/Function | Purpose |
|---|---|---|
| `config.py` | `validate_parameters()` | Validates the `parameters` dict: checks required keys, rejects unknown methods, fills optional keys with defaults |
| `hamiltonian.py` | `Hamiltonian` | Builds AO integrals (T, V, S, ERI) via Psi4 MintsHelper; handles field perturbations |
| `hf_wfn.py` | `hf_wfn` | RHF SCF with DIIS; stores `C` (MO coefficients) and `eps` (orbital energies) |
| `mp2_wfn.py` | `mp2_wfn` | MP2 energy and amplitudes (spatial and spin-orbital bases) |
| `ci_wfn.py` | `ci_wfn` | CID and CISD iterative solvers with DIIS |
| `cc_wfn.py` | `cc_wfn` | Coupled cluster wavefunction object. `solve_CCSD_SO()` is a spin-orbital CCSD solver with DIIS, optionally solving the Lambda amplitudes; both the T and Lambda residuals are factored into intermediates (see CCSD notes below) |
| `energy.py` | `energy()` | Unified entry point: builds `Hamiltonian`, runs SCF, dispatches to post-HF; returns `(E_list, T_list, C, basis)` where `T_list = [t0, t1, t2, l1, l2]` |
| `fin_diff.py` | `finite_difference` | Numerical Hessian, APTs, AATs, and velocity-gauge APTs (`compute_VG_APT`) via central differences |
| `aats.py` | `AAT` | Finite-difference atomic axial tensors; handles phase alignment between displaced wavefunctions |
| `vg_apts_fd.py` | `VG_APT` | Finite-difference velocity-gauge atomic polar tensors for RHF, MP2, CID, CISD; mirrors `aats.py:AAT` (same phase-alignment machinery, momentum rather than magnetic perturbation) |
| `parallel.py` | `compute_parallel_hessian()`, `compute_parallel_apt()`, `compute_parallel_aats()`, `compute_parallel_vg_apt()` | Parallelizes finite-difference property computations using `multiprocessing`; Hessian/APT use spawned worker processes returning scalars; AAT and VG APT run SCF serially then distribute tensor contractions |
| `analytic_base.py` | `AnalyticDerivative` | Base class for all analytic derivative objects: runs one RHF SCF on construction, provides `_setup_mo_basis()`, `_build_cphf_A()`, and `_solve_cphf()` (vectorized batched CPHF solve for all perturbation directions at once) |
| `analytic_hessian.py` | `analytic_derivative(AnalyticDerivative)` | Analytic nuclear Hessian for RHF wavefunctions (two implementations: `compute_RHF_Hessian` and `compute_RHF_Hessian_opt`) |
| `analytic_aats.py` | `analytic_derivative(AnalyticDerivative)` | Analytic atomic axial tensors (magnetic field response) for RHF, MP2, CID, CISD |
| `analytic_apts.py` | `analytic_derivative(AnalyticDerivative)` | Analytic atomic polar tensors in length gauge (LG) and velocity gauge (VG) for RHF, MP2, CID, CISD |
| `ps_analytic_hessian.py` | `analytic_derivative` | Phase-space momentum Hessian for kinetic-energy-weighted frequencies |
| `freq.py` | `frequency` | Computes vibrational frequencies from position and/or momentum Hessians |
| `vcd.py` | `vcd` | Combines Hessian + APTs + AATs to produce VCD spectral intensities |
| `integrals.py` | `one_electron_integral()` | Manual (non-Psi4) computation of AO integrals (overlap, dipole, nabla, angular momentum, kinetic, potential) |
| `utils.py` | — | DIIS solvers, MO/SO integral transforms (`compute_F_MO`, `compute_ERI_MO`, `compute_F_SO`, `compute_ERI_SO`), MO overlap and phase correction for finite differences, `total_energy()`, `get_slices()`, `line_shape()` (standalone Lorentzian broadening helper for users plotting spectra) |

### Typical VCD Calculation Flow

```python
# 1. Reference energy
E_list, T_list, C, basis = apyib.energy.energy(parameters)

# 2. Analytic Hessian
hess = apyib.analytic_hessian.analytic_derivative(parameters)
Hessian = hess.compute_RHF_Hessian(orbitals='non-canonical')

# 3. Analytic APTs (length and/or velocity gauge)
apts = apyib.analytic_apts.analytic_derivative(parameters)
P_LG = apts.compute_RHF_APTs_LG(orbitals='non-canonical')

# 4. Analytic AATs
aats = apyib.analytic_aats.analytic_derivative(parameters)
I = aats.compute_RHF_AATs(orbitals='non-canonical')

# 5. VCD spectrum
vcd = apyib.vcd.vcd(parameters)
w, D_rr, R_rl = vcd.compute_vcd_from_input(Hessian, P_LG, I)
```

### Spin-Orbital vs. Spatial-Orbital

Methods suffixed `_SO` (e.g., `MP2_SO`, `CID_SO`, `CCSD_SO`) operate in the spin-orbital basis. The `utils.py` functions `compute_F_SO` and `compute_ERI_SO` transform from the MO basis. The non-`_SO` variants use the spatial (restricted) orbital basis.

### Coupled Cluster (CCSD)

`cc_wfn.solve_CCSD_SO(print_level=0, solving_lambda=False)` is a spin-orbital CCSD
solver validated against Psi4 to ~1e-13 for H2O2/STO-3G and H2O/6-31G*, with and
without frozen core (`test_027_CCSD_SO.py`).

The t and Lambda amplitudes are validated against CFOUR through their
phase-invariant norms, which is what survives differing MO sign conventions between
codes. For H2O2/STO-3G: `||T1_AA||` 0.0272672764 vs 0.0272672765, `||T2_AB||`
0.2611128380 vs 0.2611128386, `||L1_AA||` 0.0253522480 vs 0.0253522486, `||L2_AB||`
0.2534103541 vs 0.2534103541 -- all at CFOUR's own `CC_CONV=10` threshold. The ZMAT
needs `SYMMETRY=OFF` so CFOUR does not reorient the molecule, `PROP=FIRST_ORDER` to
run `xlambda`, and `PRINT=5` to print the norms; the first norms printed are the
initial guess (CFOUR also starts from `l = t`), so read the last set.

**Staging.** Spin-orbital CCSD (done) -> Lambda amplitudes (done) -> Lambda
intermediates (done) -> spin-adapted CCSD. The near-term goal is CCSD-level AATs
by finite difference, which is what the Lambda amplitudes are for.

**Structure.** The T residuals are factored into the one-particle intermediates
`F_ae`, `F_mi`, `F_me` and the two-particle `W_mnij`, `W_abef`, `W_mbej`, each built
by its own method so a caller can take only what it needs. `_residuals_SO` builds
them and returns both T residuals; the doubles residual folds onto representatives
carrying `P(i,j)` and `P(a,b)`. The Lambda residuals are factored the same way, with
`_lambda_residuals_SO` as the matching entry point.

Two conventions differ from the published forms and must not be "corrected":

- `F_ae` and `F_mi` keep the diagonal of `f`. The published `(1 - delta)` versions
  move that diagonal to the left-hand side and solve `t * D = RHS`; this solver
  evaluates the full residual and updates `t += r / D`, so dropping the diagonal
  would double-count it.
- `W_abef` carries no tau term. The quartic tau-tau contribution is routed entirely
  through `W_mnij`, whose tau term is `1/2` rather than `1/4`. Splitting it `1/4`
  and `1/4` is equally correct but costs `O(n^2 N^4)` against `O(n^4 N^2)`.

DIIS extrapolates on the amplitude increments `dt = r / D` (equivalently
`t_new - t_old`), not on the bare residual.

**Factoring bought little here, and will not buy much more.** `opt_einsum` already
reduces the unfactored residuals from N^8 to N^6 by finding the pairwise contraction
path itself -- measured N^6.0 with the path search on, N^7.2 with it off. The
intermediates are worth having because the Lambda and density equations are written
in terms of them, not for speed. The particle-particle ladder `<ab||cd> t_ij^cd` is
the most expensive term in any arrangement and is irreducible.

**Structural difference from the CI solvers.** The CC residuals carry no
`-E * t` term. The CC amplitude equations are projective (`0 = <Phi_ij^ab|H_bar|Phi_0>`)
rather than an eigenvalue problem, and the energy is evaluated separately from the
converged amplitudes. Carrying the `ci_wfn` habit across produces a subtly wrong
answer that still converges.

**Where the equations come from.** They are generated, not hand-derived:
`~/Documents/Notes/general_hessian/eq_generator/ccsd/fapy_input.py`, run under the
`fapy` conda env against the engine at `~/src/fapy`. That input file also derives
the Lambda and response-density equations, so no new derivation work is needed for
later stages. `~/Documents/Journal_Articles/DPD_Approach_to_CCSD.pdf` (Stanton) is
the reference for the standard intermediates, and
`~/Documents/Notes/ccsd_derivation/` carries the worked derivation of the factored
Lambda equations plus the scripts that verify it.

**Bra-ket index safety (important for magnetic-field work).** Two symmetries look
alike and behave differently:

- Antisymmetry, `<pq||rs> = -<pq||sr> = -<qp||rs>` -- holds for real and complex
  orbitals. Safe to use.
- Bra-ket pair exchange, `<pq||rs> = <rs||pq>`, and `f_pq = f_qp` -- **real orbitals
  only**. For complex orbitals the two orderings are complex conjugates, so applying
  these silently introduces a conjugation error.

This is invisible in the current tests, which all run in real arithmetic where the
wrong symmetry still gives the right number. It becomes wrong the moment `F_mag` is
switched on for finite-difference AATs. fapy models this distinction directly
(`_SYMMETRY_GENERATORS` vs `_HERMITIAN_SYMMETRY` in `canonicalize.py`) and
`Problem.symmetry` defaults to `"complex"`, so the generated equations are
bra-ket-safe by construction. Preserve that when transcribing: in `cc_wfn.py`,
`+1 f(a,i)` is written `F.swapaxes(0,1)[o_,v_]` (keeping `f_ai`) and the doubles
driver `+1 g(a,b,i,j)` is written `G.swapaxes(0,2).swapaxes(1,3)[o_,o_,v_,v_]`
(keeping `<ab||ij>`). Neither may be "simplified" to `F[o_,v_]` or `G[o_,o_,v_,v_]`.

When restructuring the residuals, verify elementwise against the current
implementation using random **complex** `t1`, `t2`, `F`, and `G` -- comparing energies
in real arithmetic will not catch a bra-ket inversion.

**Lambda amplitudes.** `solving_lambda=True` solves the Lambda equations from the
converged t amplitudes. `solve_CCSD_SO` always returns a fixed 5-tuple
`(E, t1, t2, l1, l2)`, with `0` sentinels when Lambda was not requested, and
`energy()` exposes them as `T_list[3]` and `T_list[4]`. `_run_post_hf` reads the flag
as `parameters.get('solve_lambda', False)`.

The Lambda residuals were generated from the fapy terms rather than transcribed, so
no hand-copying step stands between the generator and the code. The generated form is
45 singles terms and 32 doubles representatives carrying `P(k,l)` and `P(c,d)`; both
are now factored down to **11 contractions apiece** behind eight intermediates. Since
t is fixed, the equations are linear in Lambda, and every t-dependent intermediate is
a loop invariant that `_solve_lambda_SO` builds once before the iteration; only the
two `G` intermediates depend on Lambda and are rebuilt each cycle.

**Factored Lambda equations.** The derivation is
`~/Documents/Notes/ccsd_derivation/main.tex` (content in `theory.tex`), which works
each residual in four steps from the raw fapy output: restate, relabel and reorder,
collect behind each shared Lambda amplitude, substitute. Points that matter for the
implementation:

- **Six of the eight intermediates serve both residuals, read at different indices:**
  `F~_ea`, `F~_im`, `W~_eima`, `F_ia`, `G_om`, `G_ge`. The doubles reads them as
  `F~_eb`, `F~_jm`, `W~_ejmb`, `F_jb`, `G_mj`, `G_eb`. `W~_efma` and `W~_eimn` are
  singles-only; `W~_ijmn` and `W~_efab` are doubles-only.
- `F_ia` is the T-side `F_me` at `(i,a)`, and the code calls `_F_me_SO` for it rather
  than building a second object. `<mi||ea> = <im||ae>` is antisymmetry in both pairs,
  so this is safe for complex orbitals -- unlike bra-ket exchange.
- `W~_ijmn` and `W~_efab` carry `P(mn)` and `P(ef)`, **not** a factor of 2. The
  factor-2 form contracts to the same number, because the Lambda it meets is
  antisymmetric in those indices, but the stored tensor would then lack the symmetry
  of the block it represents. `W~_efma` and `W~_eimn` do carry explicit factor-2
  terms and are correspondingly *not* antisymmetric in their summed pair. That is
  transcribed as the document has it and is settled: do not antisymmetrize them, and
  do not assume the symmetry when reusing them elsewhere.
- The two G terms in the doubles read `<mi||ab> (G_mj - sum_e t_m^e l_j^e)`. The
  `t1*l1` piece is deliberately left outside `G`. Folding it in is legitimate, but
  the augmented `G` leaves `sum <oi||ga> t_o^e t_m^g l_m^e` on the singles side at
  twice the size of the one `W~_eima` term available to absorb it, and both G terms
  want that same term -- so a merged `G` costs extra terms rather than saving any.
- The document's derivation is verified numerically against the generated equation in
  **complex** arithmetic by `~/Documents/Notes/ccsd_derivation/verify/` (see its
  README). The code is verified against the generated equation the same way, by
  `test_028`. Re-run both after editing either the document or the transcription.

Index letters in the code follow the document: the singles residual is written over
free `i` (occupied) and `a` (virtual), the doubles over `i,j` and `a,b`, with `m,n,o`
and `e,f,g` as dummies.

The Lambda drivers are `f_kc` and `<kl||cd>` -- the bra-ket transposes of the T
drivers `f_ai` and `<ab||ij>`, because Lambda projects from the left. In real
arithmetic these are the same numbers, so writing one with the T orientation passes
every real test and fails the moment `F_mag` is on.

`solve_lambda` is deliberately **not** in `config._OPTIONAL_DEFAULTS`. The parameter
dict describes the system being solved (Hamiltonian, fields, basis) plus universal
convergence criteria; `solve_lambda` selects which results are returned and is
specific to both the wavefunction and the property. `validate_parameters` does not
reject unknown keys, so the flag works while staying absent from every parameter
dict that has no use for it. Always read it with `.get`, never by direct indexing.

Note that `aats.py` normalizes with `N = 1/sqrt(t0^2 + <t1|t1> + (1/4)<t2|t2>)`, a
CI-style intermediate normalization. CCSD is not normalized that way -- its density
is the non-Hermitian `(1 + Lambda) e^-T ... e^T` form -- so the FD AAT code will need
a CC-specific branch rather than reusing that expression.

### Velocity-Gauge APTs

Velocity-gauge (VG) atomic polar tensors are available for RHF, MP2, CID, and CISD via two
independent routes, which agree to ~3e-7 on H2O/STO-3G (finite-difference truncation limit):

- **Analytic:** `analytic_apts.compute_RHF_APTs_VG(orbitals)`, and `compute_{MP2,CID,CISD}_APTs_VG(normalization, orbitals, print_level)`
- **Finite difference:** `vg_apts_fd.VG_APT`, driven by `fin_diff.compute_VG_APT(nuc_pert_strength, mom_pert_strength)`

The momentum perturbation is applied in `hamiltonian.py` as `-1j * F_mom[alpha] * mints.ao_nabla()[alpha]`,
selected by the `F_mom` parameter.

VG APTs are structurally close to AATs but differ in several places worth remembering when
editing either:

| | AAT | VG APT |
|---|---|---|
| Operator | angular momentum, `-0.5 * ao_angular_momentum()` | nabla, `-ao_nabla()` (no 1/2) |
| Prefactor | `1/(4 * nuc * mag)` | `1/(2 * nuc * mom)` |
| Nuclear term | none | adds `+Z[lambda] * delta_{alpha,beta}` |
| `dERI_dA` | pure orbital rotation | pure orbital rotation (no skeleton ERI for the field perturbation) |

CPHF sign conventions are the same as for AATs (`sign=-1, ov_sign=+1, dep_sign=-1`).

### Phase Correction

Finite-difference AAT calculations require phase alignment of MO coefficients across displaced geometries. `utils.compute_phase()` and `utils.compute_mo_overlap()` handle this. The `attic/` directory contains older standalone scripts.

### Test Data

Molecular geometries are stored in `apyib/data/molecules.py` as a `moldict` dictionary (Psi4 format strings). Tests import them via `from ..data.molecules import *`. Tests compare against Psi4 reference values and hardcoded literature values.

CCSD coverage lives in `test_027_CCSD_SO.py` (6 tests: H2O2/STO-3G and H2O/6-31G*, each with frozen core on and off, against live Psi4 plus hardcoded references; a check that the `cc_wfn` solver and the `energy()` dispatch agree; and a complex-arithmetic antisymmetry check on the doubles residual).

Lambda coverage lives in `test_028_CCSD_Lambda_SO.py` (10 tests: residual drivers, doubles antisymmetry, and linearity in Lambda, all in complex arithmetic; the factored residuals matching the generated unfactored form elementwise; `F_ia` being the T-side `F_me`; `W~_ijmn` and `W~_efab` carrying their block antisymmetry; converged Lambda annihilating its residuals with frozen core on and off; that requesting Lambda leaves the energy and t amplitudes untouched; and that the iteration converges without DIIS).

The generated unfactored residuals are kept in that test file as
`_generated_lambda_singles` / `_generated_lambda_doubles`. They are the oracle for the
factored implementation and share no algebra with it, so they are not dead code. That last test exists because DIIS extrapolates to where the error vector vanishes regardless of the step it is given, so a wrong update still reaches the right answer with DIIS on and diverges without it.

VG APT coverage lives in `test_012_VG_APT_FD.py` (finite difference), `test_014_VG_APT_parallel.py` (parallel driver), and `test_024/025/026_{MP2,CID,CISD}_VG_APT.py` (analytic, 8 cases each: H2O2/STO-3G and H2O/6-31G*, canonical/non-canonical x frozen-core on/off).

Tests marked `@pytest.mark.slow` (currently `test_023_VCD.py` and `test_015–017` analytic Hessian stubs) are excluded from the default CI run and must be opted-in explicitly (`-m slow`).

### Developer Notes

`docs/developer_notes.md` contains diagnostic and verification snippets that are intentionally kept out of production code: the SCF energy/density cross-checks removed from `hf_wfn.py`, and the CISD AAT component cross-check that was removed from `analytic_aats.py`. Consult it when debugging those subsystems.
