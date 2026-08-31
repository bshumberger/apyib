# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Velocity-gauge (VG) atomic polar tensors for MP2, CID, and CISD, extending the
  existing HF-level implementation to correlated wavefunctions and enabling VCD
  spectra at correlated levels in the velocity gauge. Implemented via two
  independent routes and cross-validated against each other: analytic
  (`analytic_apts.compute_{MP2,CID,CISD}_APTs_VG`) and finite difference
  (`vg_apts_fd.VG_APT`, driven by `fin_diff.compute_VG_APT`). The two agree to
  ~3e-7 on H2O/STO-3G at step 1e-4, the finite-difference truncation limit.
- `vg_apts_fd.py`: new `VG_APT` class, mirroring `aats.py:AAT` including the MO
  phase-alignment machinery for displaced geometries.
- `F_mom` momentum (nabla) perturbation, applied in `hamiltonian.py` via
  `mints.ao_nabla()` and given an optional default in `config.py`.
- `parallel.compute_parallel_vg_apt` for parallel finite-difference VG APTs,
  following the AAT pattern (serial SCF, parallel tensor contractions).
- Test coverage: `test_012_VG_APT_FD.py` (finite difference),
  `test_014_VG_APT_parallel.py` (parallel driver), and
  `test_024/025/026_{MP2,CID,CISD}_VG_APT.py` (analytic, 8 cases each covering
  H2O2/STO-3G and H2O/6-31G* with canonical/non-canonical orbitals and frozen
  core on/off).

### Changed
- Restored `@pytest.mark.skip` markers on the slow spin-orbital AAT cases in
  `test_013_AAT_parallel.py`.

## [1.0.0] - 2026-06-19

First stable release, following a staged modernization of the package.

### Added
- NumPy-style docstrings across the public API, and inline reference-style
  comments throughout the analytic-derivative core (CPHF machinery, Hessian,
  APT, and AAT contractions).
- `AnalyticDerivative` base class (`analytic_base.py`) providing a single shared
  RHF SCF, MO-basis setup, and a vectorized CPHF solver (`_solve_cphf`) shared
  by the analytic Hessian, APT, and AAT modules.
- Parameter validation and defaults layer (`config.py`).
- Parallel finite-difference driver (`parallel.py`): `compute_parallel_hessian`,
  `compute_parallel_apt`, and `compute_parallel_aats`.
- `utils.total_energy` helper.
- `docs/developer_notes.md` with diagnostic and verification snippets.
- Opt-in slow-test CI job and `pytest-xdist` support for parallel test runs.

### Changed
- CI runs on Ubuntu and macOS across Python 3.10–3.12, using the conda
  environment file; slow tests are excluded from the default run.
- Method dispatch in `energy.py` and `parallel.py` now raises clear errors on
  unknown methods instead of failing silently.
- CPHF solves are vectorized across all perturbation directions at once.
- Diagnostic prints in the analytic AAT and CI modules are gated behind
  `print_level`.

### Removed
- Legacy `analytic.py` (archived to `attic/`).
- Dead code: unused `solve_DIIS`, commented-out method drafts and debug blocks.

### Fixed
- CI environment activation (previously ran system Python without numpy).
