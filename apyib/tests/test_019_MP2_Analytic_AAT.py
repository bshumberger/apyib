import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_mp2_analytic_aat_h2o2_STO_3G_canonical():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0045203209,  0.0092914111,  0.0280806257],
     [ 0.099326696 , -0.1164406147,  0.3046194933],
     [-0.1691816224, -0.1866053648,  0.1294138693],
     [-0.0045203215,  0.0092914172, -0.0280806213],
     [ 0.0993267   , -0.1164406261, -0.3046194972],
     [ 0.1691815574,  0.1866053675,  0.1294138611],
     [-0.0025075901,  0.1061632799, -0.1732310166],
     [-0.0665644468, -0.1881778613,  0.640955739 ],
     [ 0.1414604078, -0.5285610823,  0.1846743587],
     [-0.002507544 ,  0.1061633078,  0.173231027 ],
     [-0.0665644769, -0.1881778547, -0.6409557401],
     [-0.1414604539,  0.5285611004,  0.184674354 ]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='canonical')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o2_STO_3G_noncanonical():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0045203209,  0.0092914111,  0.0280806257],
     [ 0.099326696 , -0.1164406147,  0.3046194933],
     [-0.1691816224, -0.1866053648,  0.1294138693],
     [-0.0045203215,  0.0092914172, -0.0280806213],
     [ 0.0993267   , -0.1164406261, -0.3046194972],
     [ 0.1691815574,  0.1866053675,  0.1294138611],
     [-0.0025075901,  0.1061632799, -0.1732310166],
     [-0.0665644468, -0.1881778613,  0.640955739 ],
     [ 0.1414604078, -0.5285610823,  0.1846743587],
     [-0.002507544 ,  0.1061633078,  0.173231027 ],
     [-0.0665644769, -0.1881778547, -0.6409557401],
     [-0.1414604539,  0.5285611004,  0.184674354 ]]) 

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='non-canonical')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o2_STO_3G_canonical_fc():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0045202942,  0.0092914181,  0.0280806887],
     [ 0.0993268744, -0.1164408873,  0.3046194463],
     [-0.1691819252, -0.1866050327,  0.1294140893],
     [-0.0045202948,  0.0092914243, -0.0280806843],
     [ 0.0993268784, -0.1164408987, -0.3046194502],
     [ 0.1691818603,  0.1866050354,  0.1294140811],
     [-0.0025109739,  0.1061860766, -0.1732631768],
     [-0.0665519894, -0.1881978196,  0.6409643484],
     [ 0.1414383445, -0.5285488815,  0.1846989613],
     [-0.0025109278,  0.1061861046,  0.1732631871],
     [-0.0665520195, -0.188197813 , -0.6409643496],
     [-0.1414383907,  0.5285488996,  0.1846989565]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='canonical')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o2_STO_3G_noncanonical_fc():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0045202942,  0.0092914181,  0.0280806887],
     [ 0.0993268744, -0.1164408873,  0.3046194463],
     [-0.1691819252, -0.1866050327,  0.1294140893],
     [-0.0045202948,  0.0092914243, -0.0280806843],
     [ 0.0993268784, -0.1164408987, -0.3046194502],
     [ 0.1691818603,  0.1866050354,  0.1294140811],
     [-0.0025109739,  0.1061860766, -0.1732631768],
     [-0.0665519894, -0.1881978196,  0.6409643484],
     [ 0.1414383445, -0.5285488815,  0.1846989613],
     [-0.0025109278,  0.1061861046,  0.1732631871],
     [-0.0665520195, -0.188197813 , -0.6409643496],
     [-0.1414383907,  0.5285488996,  0.1846989565]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o_6_31Gd_canonical():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.0996065921,  0.0000000003],
     [ 0.1439460867, -0.0000000001,  0.0000000002],
     [ 0.0000000002, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0485196329, -0.0724686573],
     [-0.0435301553, -0.          ,  0.0000000001],
     [ 0.0721521934, -0.          ,  0.0000000001],
     [ 0.0000000001,  0.0485196331,  0.0724686568],
     [-0.0435301556,  0.0000000001, -0.0000000001],
     [-0.0721521939, -0.0000000001,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o_6_31Gd_noncanonical():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.0996065921,  0.0000000003],
     [ 0.1439460867, -0.0000000001,  0.0000000002],
     [ 0.0000000002, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0485196329, -0.0724686573],
     [-0.0435301553, -0.          ,  0.0000000001],
     [ 0.0721521934, -0.          ,  0.0000000001],
     [ 0.0000000001,  0.0485196331,  0.0724686568],
     [-0.0435301556,  0.0000000001, -0.0000000001],
     [-0.0721521939, -0.0000000001,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o_6_31Gd_canonical_fc():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.0994803947,  0.0000000003],
     [ 0.1439184663, -0.0000000001,  0.0000000002],
     [ 0.0000000002, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0485202745, -0.0724691418],
     [-0.0435304593, -0.          ,  0.0000000001],
     [ 0.072152566 , -0.          ,  0.0000000001],
     [ 0.0000000001,  0.0485202747,  0.0724691413],
     [-0.0435304596,  0.0000000001, -0.0000000001],
     [-0.0721525664, -0.0000000001,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o_6_31Gd_noncanonical_fc():
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
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.0994803947,  0.0000000003],
     [ 0.1439184663, -0.0000000001,  0.0000000002],
     [ 0.0000000002, -0.0000000003,  0.0000000003],
     [-0.          ,  0.0485202745, -0.0724691418],
     [-0.0435304593, -0.          ,  0.0000000001],
     [ 0.072152566 , -0.          ,  0.0000000001],
     [ 0.0000000001,  0.0485202747,  0.0724691413],
     [-0.0435304596,  0.0000000001, -0.0000000001],
     [-0.0721525664, -0.0000000001,  0.0000000002]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_MP2_AATs(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)
