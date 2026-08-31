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

def test_mp2_analytic_vg_apt_h2o2_STO_3G_canonical():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9583646364,  0.1715615859,  0.103718249 ],
     [ 0.0768771007,  0.3208673895, -0.3479384241],
     [ 0.0428637056, -0.3433923733,  0.652763834 ],
     [ 0.9583646364,  0.1715615859, -0.103718249 ],
     [ 0.0768771007,  0.3208673895,  0.3479384241],
     [-0.0428637056,  0.3433923733,  0.652763834 ],
     [ 6.805433705 ,  0.0639385057,  0.1625133807],
     [ 0.1582394029,  7.2946780407, -0.5577972537],
     [ 0.2600286001, -0.5493335672,  7.8092016549],
     [ 6.805433705 ,  0.0639385057, -0.1625133807],
     [ 0.1582394029,  7.2946780407,  0.5577972537],
     [-0.2600286001,  0.5493335672,  7.8092016549]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o2_STO_3G_non_canonical():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9583646364,  0.1715615859,  0.103718249 ],
     [ 0.0768771007,  0.3208673895, -0.3479384241],
     [ 0.0428637056, -0.3433923733,  0.652763834 ],
     [ 0.9583646364,  0.1715615859, -0.103718249 ],
     [ 0.0768771007,  0.3208673895,  0.3479384241],
     [-0.0428637056,  0.3433923733,  0.652763834 ],
     [ 6.805433705 ,  0.0639385057,  0.1625133807],
     [ 0.1582394029,  7.2946780407, -0.5577972537],
     [ 0.2600286001, -0.5493335672,  7.8092016549],
     [ 6.805433705 ,  0.0639385057, -0.1625133807],
     [ 0.1582394029,  7.2946780407,  0.5577972537],
     [-0.2600286001,  0.5493335672,  7.8092016549]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o2_STO_3G_canonical_fc():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9583683405,  0.171561946 ,  0.1037184248],
     [ 0.0768790477,  0.3208682726, -0.3479393706],
     [ 0.0428647547, -0.3433932279,  0.6527655976],
     [ 0.9583683405,  0.171561946 , -0.1037184248],
     [ 0.0768790477,  0.3208682726,  0.3479393706],
     [-0.0428647547,  0.3433932279,  0.6527655976],
     [ 6.805384562 ,  0.0639315553,  0.1625068947],
     [ 0.1582208337,  7.2946341745, -0.557802077 ],
     [ 0.2600229338, -0.549338938 ,  7.8091631466],
     [ 6.805384562 ,  0.0639315553, -0.1625068947],
     [ 0.1582208337,  7.2946341745,  0.557802077 ],
     [-0.2600229338,  0.549338938 ,  7.8091631466]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o2_STO_3G_non_canonical_fc():
    parameters = {'geom': H2O2_manuscript,
                  'basis': 'STO-3G',
                  'method': 'MP2',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 0.9583683405,  0.171561946 ,  0.1037184248],
     [ 0.0768790477,  0.3208682726, -0.3479393706],
     [ 0.0428647547, -0.3433932279,  0.6527655976],
     [ 0.9583683405,  0.171561946 , -0.1037184248],
     [ 0.0768790477,  0.3208682726,  0.3479393706],
     [-0.0428647547,  0.3433932279,  0.6527655976],
     [ 6.805384562 ,  0.0639315553,  0.1625068947],
     [ 0.1582208337,  7.2946341745, -0.557802077 ],
     [ 0.2600229338, -0.549338938 ,  7.8091631466],
     [ 6.805384562 ,  0.0639315553, -0.1625068947],
     [ 0.1582208337,  7.2946341745,  0.557802077 ],
     [-0.2600229338,  0.549338938 ,  7.8091631466]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o_631Gs_canonical():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3799608426, 0.0, 0.0],
     [ 0.0, 2.8467261412, 0.0],
     [ 0.0, 0.0, 3.2490587396],
     [ 0.7152748310, 0.0, 0.0],
     [ 0.0, 0.4039983920, -0.2754249782],
     [ 0.0, -0.2762739547, 0.4034382989],
     [ 0.7152748310, 0.0, 0.0],
     [ 0.0, 0.4039983920, 0.2754249782],
     [ 0.0, 0.2762739547, 0.4034382989]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o_631Gs_non_canonical():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3799608426, 0.0, 0.0],
     [ 0.0, 2.8467261412, 0.0],
     [ 0.0, 0.0, 3.2490587396],
     [ 0.7152748310, 0.0, 0.0],
     [ 0.0, 0.4039983920, -0.2754249782],
     [ 0.0, -0.2762739547, 0.4034382989],
     [ 0.7152748310, 0.0, 0.0],
     [ 0.0, 0.4039983920, 0.2754249782],
     [ 0.0, 0.2762739547, 0.4034382989]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o_631Gs_canonical_fc():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3802187210, 0.0, 0.0],
     [ 0.0, 2.8468777342, 0.0],
     [ 0.0, 0.0, 3.2491791503],
     [ 0.7152797935, 0.0, 0.0],
     [ 0.0, 0.4039994912, -0.2754264600],
     [ 0.0, -0.2762774755, 0.4034407096],
     [ 0.7152797935, 0.0, 0.0],
     [ 0.0, 0.4039994912, 0.2754264600],
     [ 0.0, 0.2762774755, 0.4034407096]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)


def test_mp2_analytic_vg_apt_h2o_631Gs_non_canonical_fc():
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
                  'F_mom': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    vg_apt_ref = np.array(
    [[ 3.3802187210, 0.0, 0.0],
     [ 0.0, 2.8468777342, 0.0],
     [ 0.0, 0.0, 3.2491791503],
     [ 0.7152797935, 0.0, 0.0],
     [ 0.0, 0.4039994912, -0.2754264600],
     [ 0.0, -0.2762774755, 0.4034407096],
     [ 0.7152797935, 0.0, 0.0],
     [ 0.0, 0.4039994912, 0.2754264600],
     [ 0.0, 0.2762774755, 0.4034407096]])

    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    vg_apt = analytic_derivative.compute_MP2_APTs_VG(normalization='full', orbitals='non-canonical')

    assert(np.max(np.abs(vg_apt - vg_apt_ref)) < 1e-6)
