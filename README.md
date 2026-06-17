# apyib

[![GitHub Actions Build Status](https://github.com/bshumberger/apyib/workflows/CI/badge.svg)](https://github.com/bshumberger/apyib/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/bshumberger/apyib/branch/main/graph/badge.svg)](https://codecov.io/gh/bshumberger/apyib/branch/main)

`apyib` is a Python package for computing **Vibrational Circular Dichroism (VCD)** spectra from first principles. It implements restricted Hartree-Fock (RHF), MP2, CID, and CISD electronic structure methods with both analytic and finite-difference approaches to the molecular properties needed for VCD: nuclear Hessians, atomic polar tensors (APTs), and atomic axial tensors (AATs).

> **Note:** `apyib` is under active development, particularly the phase-space approach to VCD.

## Features

- **Electronic structure:** RHF, MP2, CID, CISD (spatial and spin-orbital bases)
- **Molecular properties:** analytic and finite-difference Hessian, APTs (length and velocity gauge), AATs
- **VCD spectrum:** dipole strengths, rotational strengths, LGOI correction
- **Phase-space approach:** momentum Hessian for kinetic-energy-weighted frequencies (experimental)
- Parallelized finite-difference AAT computation via `multiprocessing`
- Built on [Psi4](https://psicode.org/) for AO integrals and geometry handling

## Installation

Psi4 must be installed first via conda (it is not available on PyPI):

```bash
conda install psi4 -c conda-forge
```

Then clone and install `apyib` in development mode:

```bash
git clone https://github.com/bshumberger/apyib.git
cd apyib
pip install -e .
```

## Quickstart: RHF VCD spectrum

All calculations are driven by a `parameters` dictionary. The molecule geometry uses [Psi4 format](https://psicode.org/psi4manual/master/psithonmol.html).

```python
import apyib

parameters = {
    'geom': """
        O  0.000000000000  0.000000000000 -0.124038860300
        O  0.000000000000  0.000000000000  1.457505899800
        H  0.000000000000  1.756820648700 -0.748311782400
        H  0.000000000000 -1.756820648700  1.581847923100
        no_com
        no_reorient
        symmetry c1
        units bohr
    """,
    'basis': 'aug-cc-pVDZ',
    'method': 'RHF',
    'e_convergence': 1e-13,
    'd_convergence': 1e-13,
    'DIIS': True,
    'freeze_core': False,
    'F_el': [0.0, 0.0, 0.0],
    'F_mag': [0.0, 0.0, 0.0],
    'max_iterations': 120,
}

# Analytic Hessian
hess = apyib.analytic_hessian.analytic_derivative(parameters)
Hessian = hess.compute_RHF_Hessian(orbitals='non-canonical')

# Analytic APTs (length gauge)
apts = apyib.analytic_apts.analytic_derivative(parameters)
P_LG = apts.compute_RHF_APTs_LG(orbitals='non-canonical')

# Analytic AATs
aats = apyib.analytic_aats.analytic_derivative(parameters)
I = aats.compute_RHF_AATs(orbitals='non-canonical')

# VCD spectrum: returns frequencies (cm^-1), dipole strengths, rotational strengths
vcd = apyib.vcd.vcd(parameters)
frequencies, dipole_strengths, rotational_strengths = vcd.compute_vcd_from_input(Hessian, P_LG, I)
```

The `parameters['method']` key selects the electronic structure level:
`'RHF'`, `'MP2'`, `'CID'`, `'CISD'`, and spin-orbital variants `'MP2_SO'`, `'CID_SO'`, `'CISD_SO'`.

## Development

```bash
# Run the fast test suite
pytest apyib/tests/ -m "not slow"

# Run all tests (includes slow analytic Hessian tests for MP2/CID/CISD)
pytest apyib/tests/
```

## License

BSD-3-Clause. Copyright (c) 2023, Brendan M. Shumberger.

## Acknowledgements

Package structure based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
