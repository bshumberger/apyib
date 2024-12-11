import psi4
import numpy as np
import apyib
import pytest
from ..data.molecules import *

def test_mp2_analytic_aat_h2o2_STO_3G():
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
    [[-0.004520273309121,  0.009291419860507,  0.028080619275103],
     [ 0.099326691141444, -0.116440602042782,  0.304619491167353],
     [-0.169181584084974, -0.186605351925758,  0.129413864153872],
     [-0.004520273370866,  0.009291419914181, -0.02808061932783 ],
     [ 0.099326691276371, -0.116440601977733, -0.304619491044308],
     [ 0.169181584009193,  0.186605351922174,  0.129413864104849],
     [-0.002507535946562,  0.106163256744905, -0.173230975412105],
     [-0.066564444782226, -0.188177826822928,  0.640955578752423],
     [ 0.14146041573959 , -0.528560953099164,  0.184674311690071],
     [-0.002507536025536,  0.106163256684572,  0.173230975407209],
     [-0.066564444774015, -0.188177826878854, -0.640955578756481],
     [-0.141460415775133,  0.528560953133441,  0.184674311681962]])

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
    aat = analytic_derivative.compute_MP2_AATs_Canonical(normalization='full')
    print(aat)
    
    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2o2_6_31G():
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
                  'basis': '6-31G',
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
    [[ 0.006084531992518,  0.024378388699783,  0.006512738966511],
     [ 0.03856527519098 , -0.15449997968007 ,  0.315173109118805],
     [-0.069350385901883, -0.165654821261417,  0.157855713857352],
     [ 0.006084531867039,  0.024378388619465, -0.006512738296499],
     [ 0.038565275179397, -0.154499979933166, -0.315173108630279],
     [ 0.069350385868076,  0.165654821418878,  0.157855713160077],
     [-0.013664365090109,  0.113176402036762, -0.15086604117448 ],
     [-0.044975804481483, -0.055355716265228,  1.158542517444472],
     [ 0.101487129369685, -1.088900311878178,  0.05973857856682 ],
     [-0.013664365119997,  0.113176402988942,  0.15086604088429 ],
     [-0.044975804497884, -0.055355717164917, -1.158542519562915],
     [-0.101487129491868,  1.088900311043078,  0.05973858031212 ]])

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
    aat = analytic_derivative.compute_MP2_AATs_Canonical(normalization='full')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

#@pytest.mark.skip(reason="Not ready.")
def test_mp2_analytic_aat_h2o_6_31Gd():
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
    [[ 0.000000000001089, -0.099606574518057, -0.000000000003396],
     [ 0.143946055651422, -0.000000000014893,  0.000000000014849],
     [ 0.000000000007203, -0.000000000016546, -0.000000000012659],
     [-0.00000000000052 ,  0.048519633637174, -0.072468679306277],
     [-0.043530153991211, -0.000000000005254, -0.000000000000343],
     [ 0.072152194016908,  0.000000000017786, -0.000000000017121],
     [ 0.000000000000758,  0.048519633625387,  0.072468679333075],
     [-0.04353015395037 ,  0.00000000000813 ,  0.000000000005554],
     [-0.072152193976472, -0.000000000021304,  0.000000000010594]])

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
    aat = analytic_derivative.compute_MP2_AATs_Canonical(normalization='full')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-6)

def test_mp2_analytic_aat_h2_2_cc_pVDZ():
    # Set parameters for the calculation.
    H2_2_manuscript = """
    H
    H 1 0.75
    H 2 1.5 1 90.0
    H 3 0.75 2 90.0 1 60.0
    no_com
    no_reorient
    symmetry c1
    """

    parameters = {'geom': H2_2_manuscript,
                  'basis': 'cc-pVDZ',
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
    [[-0.078874531987006, -0.008599631412812,  0.362976609957579],
     [ 0.031339758042965,  0.009950614376339,  0.073174072735206],
     [-0.527500952388735, -0.137850027501267,  0.085347752147536],
     [-0.053706477301491, -0.025412530262608,  0.341287084983122],
     [-0.023637906842831, -0.008864350172209,  0.108857136096826],
     [-0.521843135618594, -0.123200994663625,  0.078051199643412],
     [-0.053706477437064, -0.025412530296756, -0.341287085030863],
     [-0.023637906989627, -0.008864350203852, -0.108857136101903],
     [ 0.521843135756329,  0.123200994695032,  0.078051199609682],
     [-0.078874532164016, -0.008599631460046, -0.36297661002465 ],
     [ 0.03133975780901 ,  0.009950614319873, -0.073174072739742],
     [ 0.527500952482876,  0.137850027526278,  0.08534775218643 ]])

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
    aat = analytic_derivative.compute_MP2_AATs_Canonical(normalization='full')
    print(aat)

    assert(np.max(np.abs(aat-aat_ref)) < 1e-7)
