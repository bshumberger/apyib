# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`apyib` is a Python library for quantum chemical calculations, focused on computing Vibrational Circular Dichroism (VCD) spectra. It implements RHF, MP2, CID, and CISD methods with both finite-difference and analytic derivative approaches for molecular properties (Hessian, APTs, AATs).

**Core dependency:** [Psi4](https://psicode.org/) provides AO integrals, basis set handling, and geometry management. All calculations require a working Psi4 installation. Other key dependencies: `numpy`, `opt_einsum`, `scipy`.

## Commands

```bash
# Development install
pip install -e .

# Run all tests
pytest apyib/tests/

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
    'max_iterations': 120,
    # Optional:
    'isotopes': {atom_idx: mass},  # Isotopic substitutions
    'P_nuc': ...,                  # Nuclear momentum (phase-space)
}
```

### Module Responsibilities

| Module | Class/Function | Purpose |
|---|---|---|
| `hamiltonian.py` | `Hamiltonian` | Builds AO integrals (T, V, S, ERI) via Psi4 MintsHelper; handles field perturbations |
| `hf_wfn.py` | `hf_wfn` | RHF SCF with DIIS; stores `C` (MO coefficients) and `eps` (orbital energies) |
| `mp2_wfn.py` | `mp2_wfn` | MP2 energy and amplitudes (spatial and spin-orbital bases) |
| `ci_wfn.py` | `ci_wfn` | CID and CISD iterative solvers with DIIS |
| `energy.py` | `energy()` | Unified entry point: builds `Hamiltonian`, runs SCF, dispatches to post-HF; returns `(E_list, T_list, C, basis)` |
| `fin_diff.py` | `finite_difference` | Numerical Hessian, APTs, AATs via central differences |
| `aats.py` | `AAT` | Finite-difference atomic axial tensors; handles phase alignment between displaced wavefunctions |
| `parallel.py` | `compute_parallel_aats()` | Parallelizes AAT computation over atoms/directions using `multiprocessing` |
| `analytic_hessian.py` | `analytic_derivative` | Analytic nuclear Hessian for RHF, MP2, CID, CISD |
| `analytic_aats.py` | `analytic_derivative` | Analytic atomic axial tensors (magnetic field response) |
| `analytic_apts.py` | `analytic_derivative` | Analytic atomic polar tensors in length gauge (LG) and velocity gauge (VG) |
| `ps_analytic_hessian.py` | `analytic_derivative` | Phase-space momentum Hessian for kinetic-energy-weighted frequencies |
| `freq.py` | `frequency` | Computes vibrational frequencies from position and/or momentum Hessians |
| `vcd.py` | `vcd` | Combines Hessian + APTs + AATs to produce VCD spectral intensities |
| `integrals.py` | `one_electron_integral()` | Manual (non-Psi4) computation of AO integrals (overlap, dipole, nabla, angular momentum, kinetic, potential) |
| `utils.py` | — | DIIS solvers, MO/SO integral transforms (`compute_F_MO`, `compute_ERI_MO`, `compute_F_SO`, `compute_ERI_SO`), MO overlap and phase correction for finite differences |

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

### Phase Correction

Finite-difference AAT calculations require phase alignment of MO coefficients across displaced geometries. `utils.compute_phase()` and `utils.compute_mo_overlap()` handle this. The `attic/` directory contains older standalone scripts.

### Test Data

Molecular geometries are stored in `apyib/data/molecules.py` as a `moldict` dictionary (Psi4 format strings). Tests import them via `from ..data.molecules import *`. Tests compare against Psi4 reference values and hardcoded literature values.
