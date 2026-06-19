# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
