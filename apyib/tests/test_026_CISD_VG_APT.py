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

def test_cisd_analytic_vg_apt_h2o2_STO_3G_canonical():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9711371423,  0.1846547768,  0.1119866764],
     [ 0.0992806101,  0.287825747 , -0.3602657776],
     [ 0.0575825122, -0.3612996864,  0.6387793444],
     [ 0.9711371423,  0.1846547768, -0.1119866764],
     [ 0.0992806101,  0.287825747 ,  0.3602657776],
     [-0.0575825122,  0.3612996864,  0.6387793444],
     [ 6.8172269923,  0.0447934614,  0.1492941609],
     [ 0.1296855361,  7.3360671113, -0.5452496888],
     [ 0.2262608294, -0.5297879222,  7.8285616144],
     [ 6.8172269923,  0.0447934614, -0.1492941609],
     [ 0.1296855361,  7.3360671113,  0.5452496888],
     [-0.2262608294,  0.5297879222,  7.8285616144]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o2_STO_3G_non_canonical():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9711371423,  0.1846547768,  0.1119866764],
     [ 0.0992806101,  0.287825747 , -0.3602657776],
     [ 0.0575825122, -0.3612996864,  0.6387793444],
     [ 0.9711371423,  0.1846547768, -0.1119866764],
     [ 0.0992806101,  0.287825747 ,  0.3602657776],
     [-0.0575825122,  0.3612996864,  0.6387793444],
     [ 6.8172269923,  0.0447934614,  0.1492941609],
     [ 0.1296855361,  7.3360671113, -0.5452496888],
     [ 0.2262608294, -0.5297879222,  7.8285616144],
     [ 6.8172269923,  0.0447934614, -0.1492941609],
     [ 0.1296855361,  7.3360671113,  0.5452496888],
     [-0.2262608294,  0.5297879222,  7.8285616144]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o2_STO_3G_canonical_fc():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9711654998,  0.1846681243,  0.1120022625],
     [ 0.0993017   ,  0.2878369473, -0.3602514291],
     [ 0.0575918008, -0.3612872774,  0.6387739088],
     [ 0.9711654998,  0.1846681243, -0.1120022625],
     [ 0.0993017   ,  0.2878369473,  0.3602514291],
     [-0.0575918008,  0.3612872774,  0.6387739088],
     [ 6.8173244621,  0.0447833993,  0.1492705591],
     [ 0.1296657694,  7.3361450221, -0.5452487772],
     [ 0.2262202391, -0.5297704016,  7.8286092018],
     [ 6.8173244621,  0.0447833993, -0.1492705591],
     [ 0.1296657694,  7.3361450221,  0.5452487772],
     [-0.2262202391,  0.5297704016,  7.8286092018]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o2_STO_3G_non_canonical_fc():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'CISD',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9711654998,  0.1846681243,  0.1120022625],
     [ 0.0993017   ,  0.2878369473, -0.3602514291],
     [ 0.0575918008, -0.3612872774,  0.6387739088],
     [ 0.9711654998,  0.1846681243, -0.1120022625],
     [ 0.0993017   ,  0.2878369473,  0.3602514291],
     [-0.0575918008,  0.3612872774,  0.6387739088],
     [ 6.8173244621,  0.0447833993,  0.1492705591],
     [ 0.1296657694,  7.3361450221, -0.5452487772],
     [ 0.2262202391, -0.5297704016,  7.8286092018],
     [ 6.8173244621,  0.0447833993, -0.1492705591],
     [ 0.1296657694,  7.3361450221,  0.5452487772],
     [-0.2262202391,  0.5297704016,  7.8286092018]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o_631Gs_canonical():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.4652564110, 0.0, 0.0],
     [ 0.0, 2.9368936002, 0.0],
     [ 0.0, 0.0, 3.3611330108],
     [ 0.7136446675, 0.0, 0.0],
     [ 0.0, 0.3996557275, -0.2988531674],
     [ 0.0, -0.2866885724, 0.3865933280],
     [ 0.7136446675, 0.0, 0.0],
     [ 0.0, 0.3996557275, 0.2988531674],
     [ 0.0, 0.2866885724, 0.3865933280]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o_631Gs_non_canonical():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.4652564110, 0.0, 0.0],
     [ 0.0, 2.9368936002, 0.0],
     [ 0.0, 0.0, 3.3611330108],
     [ 0.7136446675, 0.0, 0.0],
     [ 0.0, 0.3996557275, -0.2988531674],
     [ 0.0, -0.2866885724, 0.3865933280],
     [ 0.7136446675, 0.0, 0.0],
     [ 0.0, 0.3996557275, 0.2988531674],
     [ 0.0, 0.2866885724, 0.3865933280]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o_631Gs_canonical_fc():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.4667839855, 0.0, 0.0],
     [ 0.0, 2.9383623975, 0.0],
     [ 0.0, 0.0, 3.3624757498],
     [ 0.7135085494, 0.0, 0.0],
     [ 0.0, 0.3994146407, -0.2990953421],
     [ 0.0, -0.2868249352, 0.3863884319],
     [ 0.7135085494, 0.0, 0.0],
     [ 0.0, 0.3994146407, 0.2990953421],
     [ 0.0, 0.2868249352, 0.3863884319]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_cisd_analytic_vg_apt_h2o_631Gs_non_canonical_fc():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.4667839855, 0.0, 0.0],
     [ 0.0, 2.9383623975, 0.0],
     [ 0.0, 0.0, 3.3624757498],
     [ 0.7135085494, 0.0, 0.0],
     [ 0.0, 0.3994146407, -0.2990953421],
     [ 0.0, -0.2868249352, 0.3863884319],
     [ 0.7135085494, 0.0, 0.0],
     [ 0.0, 0.3994146407, 0.2990953421],
     [ 0.0, 0.2868249352, 0.3863884319]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_CISD_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)
