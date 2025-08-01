import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_rhf_analytic_apt_nh3_6_31Gd_canonical_LG():
    # Set parameters for the calculation.
    NH3 = """
    N  0.0000   0.0000   0.1278
    H -0.8855   1.5337  -0.5920
    H -0.8855  -1.5337  -0.5920
    H  1.7710   0.0000  -0.5920
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': NH3,
                  'basis': '6-31G*',
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Gaussian APT.
    apt_ref = np.array(
    [[-3.66363346E-01, -2.93707131E-13,  1.57820993E-07], 
     [-3.46644118E-12, -3.66374705E-01,  5.85313841E-12], 
     [-6.54958920E-06, -5.52573192E-12, -6.87096507E-01], 
     [ 1.52181990E-01,  5.20697704E-02, -4.17504201E-02],
     [ 5.20654773E-02,  9.20636205E-02,  7.23145158E-02],
     [-9.46871707E-02,  1.64007420E-01,  2.29032755E-01],
     [ 1.52181990E-01, -5.20697704E-02, -4.17504201E-02],
     [-5.20654774E-02,  9.20636205E-02, -7.23145158E-02],
     [-9.46871708E-02, -1.64007420E-01,  2.29032755E-01],
     [ 6.19993662E-02, -2.34749095E-12,  8.35006824E-02],
     [ 1.61731202E-11,  1.82247464E-01, -2.73773388E-11], 
     [ 1.89380891E-01,  1.06809688E-11,  2.29030997E-01]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    apt = analytic_derivative.compute_RHF_APTs_LG(orbitals='canonical')
    print(apt)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)

def test_rhf_analytic_apt_nh3_6_31Gd_noncanonical_LG():
    # Set parameters for the calculation.
    NH3 = """ 
    N  0.0000   0.0000   0.1278
    H -0.8855   1.5337  -0.5920
    H -0.8855  -1.5337  -0.5920
    H  1.7710   0.0000  -0.5920
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': NH3,
                  'basis': '6-31G*',
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Gaussian APT.
    apt_ref = np.array(
    [[-3.66363346E-01, -2.93707131E-13,  1.57820993E-07], 
     [-3.46644118E-12, -3.66374705E-01,  5.85313841E-12], 
     [-6.54958920E-06, -5.52573192E-12, -6.87096507E-01], 
     [ 1.52181990E-01,  5.20697704E-02, -4.17504201E-02],
     [ 5.20654773E-02,  9.20636205E-02,  7.23145158E-02],
     [-9.46871707E-02,  1.64007420E-01,  2.29032755E-01],
     [ 1.52181990E-01, -5.20697704E-02, -4.17504201E-02],
     [-5.20654774E-02,  9.20636205E-02, -7.23145158E-02],
     [-9.46871708E-02, -1.64007420E-01,  2.29032755E-01],
     [ 6.19993662E-02, -2.34749095E-12,  8.35006824E-02],
     [ 1.61731202E-11,  1.82247464E-01, -2.73773388E-11], 
     [ 1.89380891E-01,  1.06809688E-11,  2.29030997E-01]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    apt = analytic_derivative.compute_RHF_APTs_LG(orbitals='non-canonical')
    print(apt)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)

def test_rhf_analytic_apt_nh3_6_31Gd_canonical_VG():
    # Set parameters for the calculation.
    NH3 = """ 
    N  0.0000   0.0000   0.1278
    H -0.8855   1.5337  -0.5920
    H -0.8855  -1.5337  -0.5920
    H  1.7710   0.0000  -0.5920
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': NH3,
                  'basis': '6-31G*',
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference APT. Values match literature reported out to third decimal place.
    apt_ref = np.array(
    [[ 2.7910108409, -0.          , -0.0000037899],
     [-0.          ,  2.7910214664, -0.          ],
     [-0.0000127205, -0.          ,  3.3549752617],
     [ 0.4952002455,  0.2127392008, -0.1422540287],
     [ 0.2127316532,  0.2495690578,  0.2463863734],
     [-0.1777576447,  0.3078900181,  0.5821812112],
     [ 0.4952002455, -0.2127392008, -0.1422540287],
     [-0.2127316532,  0.2495690578, -0.2463863734],
     [-0.1777576447, -0.3078900181,  0.5821812112],
     [ 0.1267369625, -0.          ,  0.2845040867],
     [-0.          ,  0.618035072 ,  0.          ],
     [ 0.3555202494,  0.          ,  0.5821833986]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    apt = analytic_derivative.compute_RHF_APTs_VG(orbitals='canonical')
    print(apt)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)

def test_rhf_analytic_apt_nh3_6_31Gd_noncanonical_VG():
    # Set parameters for the calculation.
    NH3 = """ 
    N  0.0000   0.0000   0.1278
    H -0.8855   1.5337  -0.5920
    H -0.8855  -1.5337  -0.5920
    H  1.7710   0.0000  -0.5920
    no_com
    no_reorient
    symmetry c1
    units bohr
    """

    parameters = {'geom': NH3,
                  'basis': '6-31G*',
                  'method': 'RHF',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'DIIS': True,
                  'freeze_core': False,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference APT. Values match literature reported out to third decimal place.
    apt_ref = np.array(
    [[ 2.7910108409, -0.          , -0.0000037899],
     [-0.          ,  2.7910214664, -0.          ],   
     [-0.0000127205, -0.          ,  3.3549752617],
     [ 0.4952002455,  0.2127392008, -0.1422540287],
     [ 0.2127316532,  0.2495690578,  0.2463863734],
     [-0.1777576447,  0.3078900181,  0.5821812112],
     [ 0.4952002455, -0.2127392008, -0.1422540287],
     [-0.2127316532,  0.2495690578, -0.2463863734],
     [-0.1777576447, -0.3078900181,  0.5821812112],
     [ 0.1267369625, -0.          ,  0.2845040867],
     [-0.          ,  0.618035072 ,  0.          ],   
     [ 0.3555202494,  0.          ,  0.5821833986]])

    print("Molecule: ", parameters['geom'])
    print("Basis: ", parameters['basis'])
    print("Theory: ", parameters['method'])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)
    E_tot = E_list[0] + E_list[1] + E_list[2]
    print(E_tot)

    # Compute analytic AATs using apyib.
    analytic_derivative = apyib.analytic_apts.analytic_derivative(parameters)
    apt = analytic_derivative.compute_RHF_APTs_VG(orbitals='non-canonical')
    print(apt)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)
