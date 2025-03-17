import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_cisd_analytic_aat_h2o2_STO_3G_canonical():
    # Set parameters for the calculation.
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

    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0068049205,  0.0213593292,  0.0157638638],
     [ 0.1059075798, -0.1346215855,  0.3154829851],
     [-0.1816121669, -0.1870490991,  0.1457004062],
     [-0.0068049172,  0.0213593396, -0.0157638621],
     [ 0.1059075875, -0.1346215787, -0.3154829816],
     [ 0.1816121034,  0.1870491027,  0.1457004024],
     [ 0.000604592 ,  0.092507305 , -0.1257420975],
     [-0.0723391342, -0.1671952983,  0.6366666857],
     [ 0.1734247032, -0.5586889001,  0.1657068017],
     [ 0.000604639 ,  0.0925073322,  0.1257421107],
     [-0.0723391636, -0.1671952941, -0.6366666901],
     [-0.1734247527,  0.558688918 ,  0.1657067989]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='canonical')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o2_STO_3G_noncanonical():
    # Set parameters for the calculation.
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

    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0068049205,  0.0213593292,  0.0157638638],
     [ 0.1059075798, -0.1346215855,  0.3154829851],
     [-0.1816121669, -0.1870490991,  0.1457004062],
     [-0.0068049172,  0.0213593396, -0.0157638621],
     [ 0.1059075875, -0.1346215787, -0.3154829816],
     [ 0.1816121034,  0.1870491027,  0.1457004024],
     [ 0.000604592 ,  0.092507305 , -0.1257420975],
     [-0.0723391342, -0.1671952983,  0.6366666857],
     [ 0.1734247032, -0.5586889001,  0.1657068017],
     [ 0.000604639 ,  0.0925073322,  0.1257421107],
     [-0.0723391636, -0.1671952941, -0.6366666901],
     [-0.1734247527,  0.558688918 ,  0.1657067989]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='non-canonical')
    print(aat)
        
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o2_STO_3G_canonical_fc():
    # Set parameters for the calculation.
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

    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0068036808,  0.0213660472,  0.015759175 ],
     [ 0.1059122212, -0.1346189372,  0.3154831852],
     [-0.1816210036, -0.1870531658,  0.1456979231],
     [-0.006803676 ,  0.0213660586, -0.0157591705],
     [ 0.1059122263, -0.1346189294, -0.3154831852],
     [ 0.1816209399,  0.18705317  ,  0.1456979166],
     [ 0.0006036163,  0.0925366607, -0.1258237748],
     [-0.0723458891, -0.1671863347,  0.6367516767],
     [ 0.1734516158, -0.5588094314,  0.1656973648],
     [ 0.000603663 ,  0.0925366905,  0.1258237846],
     [-0.0723459201, -0.1671863314, -0.6367516783],
     [-0.1734516651,  0.5588094476,  0.1656973645]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='canonical')
    print(aat)
        
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o2_STO_3G_noncanonical_fc():
    # Set parameters for the calculation.
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

    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0068036808,  0.0213660472,  0.015759175 ],
     [ 0.1059122212, -0.1346189372,  0.3154831852],
     [-0.1816210036, -0.1870531658,  0.1456979231],
     [-0.006803676 ,  0.0213660586, -0.0157591705],
     [ 0.1059122263, -0.1346189294, -0.3154831852],
     [ 0.1816209399,  0.18705317  ,  0.1456979166],
     [ 0.0006036163,  0.0925366607, -0.1258237748],
     [-0.0723458891, -0.1671863347,  0.6367516767],
     [ 0.1734516158, -0.5588094314,  0.1656973648],
     [ 0.000603663 ,  0.0925366905,  0.1258237846],
     [-0.0723459201, -0.1671863314, -0.6367516783],
     [-0.1734516651,  0.5588094476,  0.1656973645]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)  

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o_6_31Gd_canonical():
    # Set parameters for the calculation.
    H2O_manuscript = """
    O -0.000000000000000   0.000000000000000   0.128444410656440
    H  0.000000000000000  -1.415531238764228  -1.019253001167221
    H  0.000000000000000   1.415531238764228  -1.019253001167221
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': H2O_manuscript,
                  'basis': '6-31G*',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.1024831801, -0.          ],
     [ 0.1471324344, -0.0000000003,  0.0000000002],
     [ 0.0000000004, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0517931851, -0.0731986732],
     [-0.0439551554, -0.0000000001,  0.0000000001],
     [ 0.0735027593, -0.          ,  0.          ],
     [ 0.          ,  0.051793185 ,  0.0731986723],
     [-0.0439551562,  0.0000000001, -0.          ],
     [-0.0735027604, -0.0000000003,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o_6_31Gd_noncanonical():
    # Set parameters for the calculation.
    H2O_manuscript = """
    O -0.000000000000000   0.000000000000000   0.128444410656440
    H  0.000000000000000  -1.415531238764228  -1.019253001167221
    H  0.000000000000000   1.415531238764228  -1.019253001167221
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': H2O_manuscript,
                  'basis': '6-31G*',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.1024831801, -0.          ],
     [ 0.1471324344, -0.0000000003,  0.0000000002],
     [ 0.0000000004, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0517931851, -0.0731986732],
     [-0.0439551554, -0.0000000001,  0.0000000001],
     [ 0.0735027593, -0.          ,  0.          ], 
     [ 0.          ,  0.051793185 ,  0.0731986723],
     [-0.0439551562,  0.0000000001, -0.          ], 
     [-0.0735027604, -0.0000000003,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o_6_31Gd_canonical_fc():
    # Set parameters for the calculation.
    H2O_manuscript = """
    O -0.000000000000000   0.000000000000000   0.128444410656440
    H  0.000000000000000  -1.415531238764228  -1.019253001167221
    H  0.000000000000000   1.415531238764228  -1.019253001167221
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': H2O_manuscript,
                  'basis': '6-31G*',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.1024426053, -0.          ],
     [ 0.1471371316, -0.0000000003,  0.0000000002],
     [ 0.0000000004, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0518641327, -0.0732279461],
     [-0.0439850899, -0.0000000001,  0.0000000001],
     [ 0.0735242518, -0.          ,  0.          ],
     [ 0.          ,  0.0518641326,  0.0732279452],
     [-0.0439850894,  0.0000000001, -0.          ],
     [-0.0735242523, -0.0000000003,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cisd_analytic_aat_h2o_6_31Gd_noncanonical_fc():
    # Set parameters for the calculation.
    H2O_manuscript = """
    O -0.000000000000000   0.000000000000000   0.128444410656440
    H  0.000000000000000  -1.415531238764228  -1.019253001167221
    H  0.000000000000000   1.415531238764228  -1.019253001167221
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': H2O_manuscript,
                  'basis': '6-31G*',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.1024426053, -0.          ],
     [ 0.1471371316, -0.0000000003,  0.0000000002],
     [ 0.0000000004, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0518641327, -0.0732279461],
     [-0.0439850899, -0.0000000001,  0.0000000001],
     [ 0.0735242518, -0.          ,  0.          ],
     [ 0.          ,  0.0518641326,  0.0732279452],
     [-0.0439850894,  0.0000000001, -0.          ],
     [-0.0735242523, -0.0000000003,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    H = apyib.hamiltonian.Hamiltonian(parameters)
    wfn = apyib.hf_wfn.hf_wfn(H)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CISD_AATs_Canonical(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)
