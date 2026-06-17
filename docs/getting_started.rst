Getting Started
===============

Installation
------------

Psi4 must be installed via conda before installing ``apyib`` (it is not available on PyPI):

.. code-block:: bash

   conda install psi4 -c conda-forge

Then clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/bshumberger/apyib.git
   cd apyib
   pip install -e .

Running the Tests
-----------------

.. code-block:: bash

   # Fast test suite (excludes slow analytic Hessian tests)
   pytest apyib/tests/ -m "not slow"

   # Full test suite
   pytest apyib/tests/

Parameters Dictionary
---------------------

All calculations are controlled by a ``parameters`` dictionary:

.. code-block:: python

   parameters = {
       'geom': '<psi4-format geometry string>',
       'basis': 'aug-cc-pVDZ',
       'method': 'RHF',          # RHF, MP2, CID, CISD, MP2_SO, CID_SO, CISD_SO
       'e_convergence': 1e-12,
       'd_convergence': 1e-12,
       'DIIS': True,
       'freeze_core': False,
       'F_el': [0.0, 0.0, 0.0],  # electric field perturbation
       'F_mag': [0.0, 0.0, 0.0], # magnetic field perturbation
       'max_iterations': 120,
       # optional:
       'isotopes': {atom_idx: mass},
   }

Typical VCD Calculation
-----------------------

.. code-block:: python

   import apyib

   # 1. Analytic Hessian
   hess = apyib.analytic_hessian.analytic_derivative(parameters)
   Hessian = hess.compute_RHF_Hessian(orbitals='non-canonical')

   # 2. Analytic APTs (length gauge)
   apts = apyib.analytic_apts.analytic_derivative(parameters)
   P_LG = apts.compute_RHF_APTs_LG(orbitals='non-canonical')

   # 3. Analytic AATs
   aats = apyib.analytic_aats.analytic_derivative(parameters)
   I = aats.compute_RHF_AATs(orbitals='non-canonical')

   # 4. VCD spectrum
   vcd = apyib.vcd.vcd(parameters)
   frequencies, dipole_strengths, rotational_strengths = vcd.compute_vcd_from_input(
       Hessian, P_LG, I
   )
