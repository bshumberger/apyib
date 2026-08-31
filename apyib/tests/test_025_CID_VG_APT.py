import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

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

def test_cid_analytic_vg_apt_h2o2_STO_3G_canonical():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.960485723 ,  0.1729117916,  0.1044998094],
     [ 0.0780534671,  0.3194567982, -0.3473784636],
     [ 0.0433425578, -0.3433999074,  0.6512233033],
     [ 0.960485723 ,  0.1729117916, -0.1044998094],
     [ 0.0780534671,  0.3194567982,  0.3473784636],
     [-0.0433425578,  0.3433999074,  0.6512233033],
     [ 6.823951675 ,  0.0608325486,  0.1586845028],
     [ 0.1547431869,  7.3051548448, -0.5546591829],
     [ 0.2568943602, -0.5450143548,  7.813765682 ],
     [ 6.823951675 ,  0.0608325486, -0.1586845028],
     [ 0.1547431869,  7.3051548448,  0.5546591829],
     [-0.2568943602,  0.5450143548,  7.813765682 ]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o2_STO_3G_non_canonical():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.960485723 ,  0.1729117916,  0.1044998094],
     [ 0.0780534671,  0.3194567982, -0.3473784636],
     [ 0.0433425578, -0.3433999074,  0.6512233033],
     [ 0.960485723 ,  0.1729117916, -0.1044998094],
     [ 0.0780534671,  0.3194567982,  0.3473784636],
     [-0.0433425578,  0.3433999074,  0.6512233033],
     [ 6.823951675 ,  0.0608325486,  0.1586845028],
     [ 0.1547431869,  7.3051548448, -0.5546591829],
     [ 0.2568943602, -0.5450143548,  7.813765682 ],
     [ 6.823951675 ,  0.0608325486, -0.1586845028],
     [ 0.1547431869,  7.3051548448,  0.5546591829],
     [-0.2568943602,  0.5450143548,  7.813765682 ]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o2_STO_3G_canonical_fc():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9604876761,  0.1729140362,  0.1045004641],
     [ 0.0780526995,  0.3194541749, -0.3473807831],
     [ 0.0433422583, -0.3434030668,  0.6512240549],
     [ 0.9604876761,  0.1729140362, -0.1045004641],
     [ 0.0780526995,  0.3194541749,  0.3473807831],
     [-0.0433422583,  0.3434030668,  0.6512240549],
     [ 6.8239789672,  0.0608304124,  0.1586825914],
     [ 0.1547422702,  7.3051704648, -0.5546452059],
     [ 0.2568927231, -0.5449996174,  7.8137681493],
     [ 6.8239789672,  0.0608304124, -0.1586825914],
     [ 0.1547422702,  7.3051704648,  0.5546452059],
     [-0.2568927231,  0.5449996174,  7.8137681493]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o2_STO_3G_non_canonical_fc():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CID',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9604876761,  0.1729140362,  0.1045004641],
     [ 0.0780526995,  0.3194541749, -0.3473807831],
     [ 0.0433422583, -0.3434030668,  0.6512240549],
     [ 0.9604876761,  0.1729140362, -0.1045004641],
     [ 0.0780526995,  0.3194541749,  0.3473807831],
     [-0.0433422583,  0.3434030668,  0.6512240549],
     [ 6.8239789672,  0.0608304124,  0.1586825914],
     [ 0.1547422702,  7.3051704648, -0.5546452059],
     [ 0.2568927231, -0.5449996174,  7.8137681493],
     [ 6.8239789672,  0.0608304124, -0.1586825914],
     [ 0.1547422702,  7.3051704648,  0.5546452059],
     [-0.2568927231,  0.5449996174,  7.8137681493]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o_631Gs_canonical():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3789122130, 0.0, 0.0],
     [ 0.0, 2.8499492271, 0.0],
     [ 0.0, 0.0, 3.2486385841],
     [ 0.7156927295, 0.0, 0.0],
     [ 0.0, 0.4029847631, -0.2762075269],
     [ 0.0, -0.2776442934, 0.4040068098],
     [ 0.7156927295, 0.0, 0.0],
     [ 0.0, 0.4029847631, 0.2762075269],
     [ 0.0, 0.2776442934, 0.4040068098]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o_631Gs_non_canonical():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3789122130, 0.0, 0.0],
     [ 0.0, 2.8499492271, 0.0],
     [ 0.0, 0.0, 3.2486385841],
     [ 0.7156927295, 0.0, 0.0],
     [ 0.0, 0.4029847631, -0.2762075269],
     [ 0.0, -0.2776442934, 0.4040068098],
     [ 0.7156927295, 0.0, 0.0],
     [ 0.0, 0.4029847631, 0.2762075269],
     [ 0.0, 0.2776442934, 0.4040068098]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o_631Gs_canonical_fc():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3791527271, 0.0, 0.0],
     [ 0.0, 2.8501831359, 0.0],
     [ 0.0, 0.0, 3.2488246382],
     [ 0.7157031952, 0.0, 0.0],
     [ 0.0, 0.4029606016, -0.2762386399],
     [ 0.0, -0.2776753462, 0.4039897995],
     [ 0.7157031952, 0.0, 0.0],
     [ 0.0, 0.4029606016, 0.2762386399],
     [ 0.0, 0.2776753462, 0.4039897995]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cid_analytic_vg_apt_h2o_631Gs_non_canonical_fc():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3791527271, 0.0, 0.0],
     [ 0.0, 2.8501831359, 0.0],
     [ 0.0, 0.0, 3.2488246382],
     [ 0.7157031952, 0.0, 0.0],
     [ 0.0, 0.4029606016, -0.2762386399],
     [ 0.0, -0.2776753462, 0.4039897995],
     [ 0.7157031952, 0.0, 0.0],
     [ 0.0, 0.4029606016, 0.2762386399],
     [ 0.0, 0.2776753462, 0.4039897995]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CID_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)
