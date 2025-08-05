import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_cid_analytic_aat_h2o2_STO_3G_canonical():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.004613098 ,  0.008121486 ,  0.030379542 ],
     [ 0.1009533191, -0.1174549201,  0.3057822731],
     [-0.1719762646, -0.1868380186,  0.1301834787],
     [-0.0046130874,  0.008121489 , -0.030379536 ],
     [ 0.1009533079, -0.1174549356, -0.3057822818],
     [ 0.1719762476,  0.1868380155,  0.130183473 ],
     [-0.0023975885,  0.1065532754, -0.1676440404],
     [-0.0681705787, -0.1850294548,  0.640710954 ],
     [ 0.1459535351, -0.5374291485,  0.1818841054],
     [-0.0023975904,  0.10655328  ,  0.1676440462],
     [-0.0681705629, -0.1850294807, -0.6407109564],
     [-0.1459535209,  0.5374291661,  0.181884105 ]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='canonical')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o2_STO_3G_noncanonical():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.004613098 ,  0.008121486 ,  0.030379542 ],
     [ 0.1009533191, -0.1174549201,  0.3057822731],
     [-0.1719762646, -0.1868380186,  0.1301834787],
     [-0.0046130874,  0.008121489 , -0.030379536 ],
     [ 0.1009533079, -0.1174549356, -0.3057822818],
     [ 0.1719762476,  0.1868380155,  0.130183473 ],
     [-0.0023975885,  0.1065532754, -0.1676440404],
     [-0.0681705787, -0.1850294548,  0.640710954 ],
     [ 0.1459535351, -0.5374291485,  0.1818841054],
     [-0.0023975904,  0.10655328  ,  0.1676440462],
     [-0.0681705629, -0.1850294807, -0.6407109564],
     [-0.1459535209,  0.5374291661,  0.181884105 ]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='non-canonical')
    print(aat)
        
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o2_STO_3G_canonical_fc():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0046129034,  0.0081199561,  0.0303809146],
     [ 0.1009522274, -0.1174566653,  0.3057813166],
     [-0.1719745327, -0.1868350161,  0.1301847753],
     [-0.0046128928,  0.0081199591, -0.0303809088],
     [ 0.1009522162, -0.1174566808, -0.3057813248],
     [ 0.1719745157,  0.186835013 ,  0.1301847699],
     [-0.0024010004,  0.1065811145, -0.1676914795],
     [-0.0681667385, -0.1850256631,  0.6407077374],
     [ 0.1459483692, -0.537424757 ,  0.1818824115],
     [-0.0024010023,  0.1065811191,  0.1676914855],
     [-0.0681667228, -0.185025689 , -0.64070774  ],
     [-0.145948355 ,  0.5374247746,  0.1818824106]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='canonical')
    print(aat)
        
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o2_STO_3G_noncanonical_fc():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[-0.0046129034,  0.0081199561,  0.0303809146],
     [ 0.1009522274, -0.1174566653,  0.3057813166],
     [-0.1719745327, -0.1868350161,  0.1301847753],
     [-0.0046128928,  0.0081199591, -0.0303809088],
     [ 0.1009522162, -0.1174566808, -0.3057813248],
     [ 0.1719745157,  0.186835013 ,  0.1301847699],
     [-0.0024010004,  0.1065811145, -0.1676914795],
     [-0.0681667385, -0.1850256631,  0.6407077374],
     [ 0.1459483692, -0.537424757 ,  0.1818824115],
     [-0.0024010023,  0.1065811191,  0.1676914855],
     [-0.0681667228, -0.185025689 , -0.64070774  ],
     [-0.145948355 ,  0.5374247746,  0.1818824106]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o_6_31Gd_canonical():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[ 0.0000000002, -0.1006299047,  0.          ],   
     [ 0.1445079517,  0.0000000001, -0.          ],  
     [ 0.0000000003, -0.0000000003,  0.0000000001],
     [-0.          ,  0.0490511921, -0.073182963 ],
     [-0.0439178602, -0.0000000001,  0.0000000003],
     [ 0.0728666997, -0.0000000001,  0.0000000003],
     [ 0.          ,  0.0490511924,  0.0731829622],
     [-0.0439178601,  0.          ,  0.0000000001],
     [-0.0728666996, -0.0000000002,  0.0000000003]]) 

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o_6_31Gd_noncanonical():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[ 0.0000000002, -0.1006299047,  0.          ],
     [ 0.1445079517,  0.0000000001, -0.          ],
     [ 0.0000000003, -0.0000000003,  0.0000000001],
     [-0.          ,  0.0490511921, -0.073182963 ],
     [-0.0439178602, -0.0000000001,  0.0000000003],
     [ 0.0728666997, -0.0000000001,  0.0000000003],
     [ 0.          ,  0.0490511924,  0.0731829622],
     [-0.0439178601,  0.          ,  0.0000000001],
     [-0.0728666996, -0.0000000002,  0.0000000003]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o_6_31Gd_canonical_fc():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[ 0.0000000002, -0.1005291889,  0.          ],
     [ 0.1444917963,  0.0000000001, -0.          ],
     [ 0.0000000003, -0.0000000003,  0.0000000001],
     [-0.          ,  0.0490549309, -0.0731865604],
     [-0.0439192339, -0.0000000001,  0.0000000003],
     [ 0.0728711415, -0.0000000001,  0.0000000003],
     [ 0.          ,  0.0490549311,  0.0731865596],
     [-0.0439192337,  0.          ,  0.0000000001],
     [-0.0728711414, -0.0000000002,  0.0000000003]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_cid_analytic_aat_h2o_6_31Gd_noncanonical_fc():
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
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting MP2 reference MagPy AAT.
    aat_ref = np.array(
    [[ 0.0000000002, -0.1005291889,  0.          ], 
     [ 0.1444917963,  0.0000000001, -0.          ],  
     [ 0.0000000003, -0.0000000003,  0.0000000001],
     [-0.          ,  0.0490549309, -0.0731865604],
     [-0.0439192339, -0.0000000001,  0.0000000003],
     [ 0.0728711415, -0.0000000001,  0.0000000003],
     [ 0.          ,  0.0490549311,  0.0731865596],
     [-0.0439192337,  0.          ,  0.0000000001],
     [-0.0728711414, -0.0000000002,  0.0000000003]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_aats.analytic_derivative(parameters)
    aat = analytic_derivative.compute_CID_AATs(normalization='full', orbitals='non-canonical')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)
