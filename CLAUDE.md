# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`apyib` is a Python library for quantum chemical calculations, focused on computing Vibrational Circular Dichroism (VCD) spectra. It implements RHF, MP2, CID, and CISD methods with both finite-difference and analytic derivative approaches for molecular properties (Hessian, APTs, AATs).

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
    'method': 'RHF',        # RHF, MP2, CID, CISD, MP2_SO, CID_SO, CISD_SO
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
| `energy.py` | `energy()` | Unified entry point: builds `Hamiltonian`, runs SCF, dispatches to post-HF; returns `(E_list, T_list, C, basis)` |
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

Methods suffixed `_SO` (e.g., `MP2_SO`, `CID_SO`) operate in the spin-orbital basis. The `utils.py` functions `compute_F_SO` and `compute_ERI_SO` transform from the MO basis. The non-`_SO` variants use the spatial (restricted) orbital basis.

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

VG APT coverage lives in `test_012_VG_APT_FD.py` (finite difference), `test_014_VG_APT_parallel.py` (parallel driver), and `test_024/025/026_{MP2,CID,CISD}_VG_APT.py` (analytic, 8 cases each: H2O2/STO-3G and H2O/6-31G*, canonical/non-canonical x frozen-core on/off).

Tests marked `@pytest.mark.slow` (currently `test_023_VCD.py` and `test_015–017` analytic Hessian stubs) are excluded from the default CI run and must be opted-in explicitly (`-m slow`).

### Developer Notes

`docs/developer_notes.md` contains diagnostic and verification snippets that are intentionally kept out of production code: the SCF energy/density cross-checks removed from `hf_wfn.py`, and the CISD AAT component cross-check that was removed from `analytic_aats.py`. Consult it when debugging those subsystems.
