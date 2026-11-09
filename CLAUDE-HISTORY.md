# CLAUDE-HISTORY.md

Code changes made with Claude. Newest first; dates absolute.

## 2026-11-09 -- Drop Python 3.10 (branch `ci/drop-python-3.10`)

- `.github/workflows/CI.yaml`: test matrix is now Python 3.11 and 3.12.
- `pyproject.toml`: `requires-python = ">=3.11"`; 3.10 classifier removed.
- `readthedocs.yml`, `docs/requirements.yaml`: docs build on Python 3.11.
- `CHANGELOG.md`: `Removed` entry under Unreleased.

Reason: the Psi4 3.10 build on conda-forge no longer imports (see CLAUDE-CURRENT.md,
2026-11-09).
