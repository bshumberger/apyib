import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_rhf_analytic_aat_h2o2_STO_3G_canonical():
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
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0044839562,  0.0086365892,  0.0287020299],
     [ 0.0982285955, -0.1160252702,  0.3048741599],
     [-0.1671868349, -0.1865840821,  0.1290566128],
     [-0.0044839574,  0.0086365959, -0.0287020255],
     [ 0.0982285992, -0.116025281 , -0.304874164 ],
     [ 0.1671867705,  0.1865840846,  0.129056605 ],
     [-0.0025650229,  0.1083434887, -0.1781884571],
     [-0.0654502672, -0.1916177004,  0.6443510323],
     [ 0.1382514099, -0.5262083642,  0.1881118373],
     [-0.0025649761,  0.1083435167,  0.1781884679],
     [-0.0654502978, -0.1916176939, -0.6443510334],
     [-0.1382514544,  0.5262083821,  0.1881118324]])

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
    aat = analytic_derivative.compute_RHF_AATs_Canonical(orbitals='canonical')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_rhf_analytic_aat_h2o2_STO_3G_noncanonical():
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
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0044839562,  0.0086365892,  0.0287020299],
     [ 0.0982285955, -0.1160252702,  0.3048741599],
     [-0.1671868349, -0.1865840821,  0.1290566128],
     [-0.0044839574,  0.0086365959, -0.0287020255],
     [ 0.0982285992, -0.116025281 , -0.304874164 ],
     [ 0.1671867705,  0.1865840846,  0.129056605 ],
     [-0.0025650229,  0.1083434887, -0.1781884571],
     [-0.0654502672, -0.1916177004,  0.6443510323],
     [ 0.1382514099, -0.5262083642,  0.1881118373],
     [-0.0025649761,  0.1083435167,  0.1781884679],
     [-0.0654502978, -0.1916176939, -0.6443510334],
     [-0.1382514544,  0.5262083821,  0.1881118324]])

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
    aat = analytic_derivative.compute_RHF_AATs_Canonical(orbitals='non-canonical')
    print(aat)
        
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_rhf_analytic_aat_h2o_6_31Gd_canonical():
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
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.0985722978,  0.0000000003],
     [ 0.1427195219, -0.0000000001,  0.0000000002],
     [ 0.0000000002, -0.0000000003,  0.0000000004],
     [-0.          ,  0.0477992504, -0.0719582567],
     [-0.043483202 , -0.          ,  0.0000000001],
     [ 0.07160764  , -0.          ,  0.0000000001],
     [ 0.0000000001,  0.0477992506,  0.0719582562],
     [-0.0434832023,  0.0000000001, -0.0000000001],
     [-0.0716076404, -0.0000000002,  0.0000000002]])

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
    aat = analytic_derivative.compute_RHF_AATs_Canonical(orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_rhf_analytic_aat_h2o_6_31Gd_noncanonical():
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
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0000000001, -0.0985722978,  0.0000000003],
     [ 0.1427195219, -0.0000000001,  0.0000000002],
     [ 0.0000000002, -0.0000000003,  0.0000000004],
     [-0.          ,  0.0477992504, -0.0719582567],
     [-0.043483202 , -0.          ,  0.0000000001],
     [ 0.07160764  , -0.          ,  0.0000000001],
     [ 0.0000000001,  0.0477992506,  0.0719582562],
     [-0.0434832023,  0.0000000001, -0.0000000001],
     [-0.0716076404, -0.0000000002,  0.0000000002]])

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
    aat = analytic_derivative.compute_RHF_AATs_Canonical(orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)
