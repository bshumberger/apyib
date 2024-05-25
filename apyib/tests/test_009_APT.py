import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_rhf_apt():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O_C4_HF"],
                  'basis': 'STO-3G',
                  'method': 'RHF',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Hessian from CFOUR HF/STO-3G optimized geometry.
    apt_ref = np.array(
    [[-0.5596822336,  0.          ,  0.          ],
     [ 0.          , -0.0311494713,  0.          ],
     [ 0.          ,  0.          ,  0.0835319998],
     [ 0.2798411169,  0.          ,  0.          ],
     [ 0.          ,  0.0155747356, -0.1570627446],
     [ 0.          , -0.2216402091, -0.0417659999],
     [ 0.2798411169,  0.          ,  0.          ],
     [ 0.          ,  0.0155747356,  0.1570627446],
     [ 0.          ,  0.2216402091, -0.0417659999]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)

    # Compute finite difference Hessian using apyib.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    apt = finite_difference.compute_APT(0.001, 0.0001)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)

def test_mp2_apt():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O_C4_MP2"],
                  'basis': 'STO-3G',
                  'method': 'MP2',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Hessian from CFOUR HF/STO-3G optimized geometry.
    apt_ref = np.array(
    [[ -0.4973007622,  0.0000000000,  0.0000000000],
     [  0.0000000000,  0.0409050306,  0.0000000000],
     [  0.0000000000,  0.0000000000,  0.1657356778],
     [  0.2486503811,  0.0000000000,  0.0000000000],
     [  0.0000000000, -0.0204525153, -0.1673618446],
     [  0.0000000000, -0.2369326775, -0.0828678389],
     [  0.2486503811,  0.0000000000,  0.0000000000],
     [  0.0000000000, -0.0204525153,  0.1673618446],
     [  0.0000000000,  0.2369326775, -0.0828678389]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)

    # Compute finite difference Hessian using apyib.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    apt = finite_difference.compute_APT(0.001, 0.0001)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)

def test_mp2_SO_apt():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O_C4_MP2"],
                  'basis': 'STO-3G',
                  'method': 'MP2_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Hessian from CFOUR HF/STO-3G optimized geometry.
    apt_ref = np.array(
    [[ -0.4973007622,  0.0000000000,  0.0000000000],
     [  0.0000000000,  0.0409050306,  0.0000000000],
     [  0.0000000000,  0.0000000000,  0.1657356778],
     [  0.2486503811,  0.0000000000,  0.0000000000],
     [  0.0000000000, -0.0204525153, -0.1673618446],
     [  0.0000000000, -0.2369326775, -0.0828678389],
     [  0.2486503811,  0.0000000000,  0.0000000000],
     [  0.0000000000, -0.0204525153,  0.1673618446],
     [  0.0000000000,  0.2369326775, -0.0828678389]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)

    # Compute finite difference Hessian using apyib.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    apt = finite_difference.compute_APT(0.001, 0.0001)

    assert(np.max(np.abs(apt-apt_ref)) < 1e-6)

def test_cid_apt():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O_C4_CID"],
                  'basis': 'STO-3G',
                  'method': 'CID',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Hessian from CFOUR HF/STO-3G optimized geometry.
    apt_ref = np.array(
    [[-4.67574414E-01,  7.05005665E-13,  0.00000000E+00],
     [ 8.40982084E-14,  1.07315520E-01, -2.31111593E-33],
     [ 8.36893079E-13,  2.35001888E-13,  1.98997646E-01],
     [ 2.33787312E-01,  3.52018780E-17,  2.17562220E-17],
     [-9.31234471E-14, -5.36578358E-02, -1.77653034E-01],
     [ 4.06138511E-13, -2.54923176E-01, -9.94990576E-02],
     [ 2.33787312E-01, -8.76562021E-34, -6.16297582E-33],
     [-9.31586489E-14, -5.36578358E-02,  1.77653034E-01],
     [-4.06107292E-13,  2.54923176E-01, -9.94990576E-02]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)

    # Compute finite difference Hessian using apyib.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    apt = finite_difference.compute_APT(0.001, 0.0001)
    
    assert(np.max(np.abs(apt-apt_ref)) < 1e-5)

def test_cid_SO_apt():
    # Set parameters for the calculation.
    parameters = {'geom': moldict["H2O_C4_CID"],
                  'basis': 'STO-3G',
                  'method': 'CID_SO',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'DIIS': True,
                  'F_el': [0.0, 0.0, 0.0],
                  'F_mag': [0.0, 0.0, 0.0],
                  'max_iterations': 120}

    # Setting RHF reference Hessian from CFOUR HF/STO-3G optimized geometry.
    apt_ref = np.array(
    [[-4.67574414E-01,  7.05005665E-13,  0.00000000E+00],
     [ 8.40982084E-14,  1.07315520E-01, -2.31111593E-33],
     [ 8.36893079E-13,  2.35001888E-13,  1.98997646E-01],
     [ 2.33787312E-01,  3.52018780E-17,  2.17562220E-17],
     [-9.31234471E-14, -5.36578358E-02, -1.77653034E-01],
     [ 4.06138511E-13, -2.54923176E-01, -9.94990576E-02],
     [ 2.33787312E-01, -8.76562021E-34, -6.16297582E-33],
     [-9.31586489E-14, -5.36578358E-02,  1.77653034E-01],
     [-4.06107292E-13,  2.54923176E-01, -9.94990576E-02]])

    # Compute energy.
    E_list, T_list, C, basis = apyib.energy.energy(parameters)

    # Compute finite difference Hessian using apyib.
    finite_difference = apyib.fin_diff.finite_difference(parameters, basis, C)
    apt = finite_difference.compute_APT(0.001, 0.0001)
    
    assert(np.max(np.abs(apt-apt_ref)) < 1e-5)

