""" Contains integrals."""

import psi4
import numpy as np
import opt_einsum as oe
import scipy.linalg as la
import math

def one_electron_integral(parameters, integral_type = 'overlap', spherical_transform=True):
    # Function for calculating the AO basis overlap integrals.
    if integral_type != 'overlap' and integral_type != 'dipole' and integral_type != 'nabla' and integral_type != 'angular momentum' and integral_type != 'kinetic':
        raise Exception("Integral type not supported. Supported one-electron integrals include overlap, dipole, nabla, angular momentum, and kinetic energy integrals.")
    if spherical_transform == True:
        raise Exception("Spherical transforms are not currently implemented.")

    # Setting up molecule, basis sets, and other parameters.
    mol = psi4.geometry(parameters['geom'])

    psi4.set_options({'basis': parameters['basis']})
    basis = psi4.core.BasisSet.build(mol, quiet=True)

    natom = mol.natom()
    R = mol.geometry().np
    COM = mol.center_of_mass()
    C = np.array([0.0,0.0,0.0]) #np.array([COM[0], COM[1], COM[2]]) #np.array([0.0,0.0,0.0])

    nshells = basis.nshell()
    nao = basis.nao()

    # Starting the loops over shells and primitives.
    if integral_type == 'overlap':
        S = np.zeros((nao, nao))
    elif integral_type == 'dipole' or integral_type == 'kinetic' or integral_type == 'nabla' or integral_type == 'angular momentum':
        S = np.zeros((3, nao, nao))
    # Creating shell object and getting the basis function index, angular momentum, number of primitives, and center associated with shell "a".
    for a in range(nshells):
        shell_a = basis.shell(a)
        idx_a = shell_a.function_index
        ang_a = shell_a.am
        nprim_a = shell_a.nprimitive
        nfunc_a = shell_a.ncartesian
        A = R[shell_a.ncenter]

        # Creating shell object and getting the basis function index, angular momentum, number of primitives, and center associated with shell "b".
        for b in range(nshells):
            shell_b = basis.shell(b)
            idx_b = shell_b.function_index
            ang_b = shell_b.am
            nprim_b = shell_b.nprimitive
            nfunc_b = shell_b.ncartesian
            B = R[shell_b.ncenter]

            #print("AM A:", ang_a, "AM B:", ang_b, "BF Index A:", idx_a, "BF Index A:", idx_b)
            R_AB = A - B
            # Setting coefficient and exponent variables for primative "pa" on shell "a".
            for pa in range(nprim_a):
                coeff_a = shell_a.coef(pa)
                exp_a = shell_a.exp(pa)

                # Setting coefficient and exponent variables for primative "pb" on shell "b".
                for pb in range(nprim_b):
                    coeff_b = shell_b.coef(pb)
                    exp_b = shell_b.exp(pb)

                    # Building required components for the initial spherical gaussian and recurrence relations.
                    p = exp_a + exp_b
                    mu = (exp_a * exp_b) / p
                    P = (exp_a * A + exp_b * B) / p
                    R_PA = P - A
                    R_PB = P - B

                    if integral_type == 'overlap':
                        size_a = ang_a + 1
                        size_b = ang_b + 1
                    if integral_type == 'dipole':
                        size_a = ang_a + 2
                        size_b = ang_b + 1
                    if integral_type == 'nabla':
                        size_a = ang_a + 1
                        size_b = ang_b + 2
                    if integral_type == 'angular momentum':
                        size_a = ang_a + 2
                        size_b = ang_b + 2
                    if integral_type == 'kinetic':
                        size_a = ang_a + 1
                        size_b = ang_b + 3
                    #S_ = np.zeros((3, ang_a+1, ang_b+1))
                    S_ = np.zeros((3, size_a, size_b))
                    # Looping over cartesian directions.
                    for alpha in range(3):
                        # Solving for the initial overlap quantity.
                        S_[alpha][0][0] = np.sqrt(math.pi / p) * math.exp(-mu * R_AB[alpha]**2)

                        # Solving the vertical recurrence. Note that if i = 0, the second term is zero by multiplication.
                        #for i in range(ang_a):
                        for i in range(size_a-1):
                            S_[alpha][i+1][0] = R_PA[alpha] * S_[alpha][i][0] + (1 / (2 * p)) * (i * S_[alpha][max(i-1, 0)][0])

                        # Solving the horizontal recurrence. Note that if i = 0 and/or j = 0, the second and/or third terms zero are zero.
                        #for i in range(ang_a+1):
                        #    for j in range(ang_b):
                        for i in range(size_a):
                            for j in range(size_b-1):
                                S_[alpha][i][j+1] = R_PB[alpha] * S_[alpha][i][j] + (1 / (2 * p)) * (i * S_[alpha][i-1][j] + j * S_[alpha][i][j-1])

                    # Looping over angular momenta to deal with the cartesian direction of non-spherical gaussians.
                    # For example, the d-type orbitals will have xx, xy, xz, yy, yz, and zz where the angular
                    # momentum is distributed over the different Cartesian directions. The loops over angular
                    # momenta allow for the different components to be accounted for in the overlap since the
                    # index function only takes you to the first component in the shell.
                    count_a = 0
                    for i_a in range(ang_a+1):
                        for j_a in range(i_a+1):
                            x_a = ang_a - i_a
                            y_a = i_a - j_a
                            z_a = j_a
                            count_b = 0
                            for i_b in range(ang_b+1):
                                for j_b in range(i_b+1):
                                    x_b = ang_b - i_b
                                    y_b = i_b - j_b
                                    z_b = j_b
                                    if integral_type == 'overlap':
                                        S[idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * S_[1][y_a][y_b] * S_[2][z_a][z_b]
                                    if integral_type == 'dipole':
                                        S[0][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * (S_[0][x_a+1][x_b] + (A[0] - C[0]) * S_[0][x_a][x_b]) * S_[1][y_a][y_b] * S_[2][z_a][z_b]
                                        S[1][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * (S_[1][y_a+1][y_b] + (A[1] - C[1]) * S_[1][y_a][y_b]) * S_[2][z_a][z_b]
                                        S[2][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * S_[1][y_a][y_b] * (S_[2][z_a+1][z_b] + (A[2] - C[2]) * S_[2][z_a][z_b])
                                    if integral_type == 'nabla':
                                        S[0][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * (x_b * S_[0][x_a][x_b-1] - 2.0 * exp_b * S_[0][x_a][x_b+1]) * S_[1][y_a][y_b] * S_[2][z_a][z_b]
                                        S[1][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * (y_b * S_[1][y_a][y_b-1] - 2.0 * exp_b * S_[1][y_a][y_b+1]) * S_[2][z_a][z_b]
                                        S[2][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * S_[1][y_a][y_b] * (z_b * S_[2][z_a][z_b-1] - 2.0 * exp_b * S_[2][z_a][z_b+1])
                                    if integral_type == 'kinetic':
                                        S[0][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * (x_b * (x_b - 1.0) * S_[0][x_a][x_b-2] - 2.0 * exp_b * x_b * S_[0][x_a][x_b] - 2.0 * exp_b * (x_b + 1) * S_[0][x_a][x_b] + 4.0 * exp_b**2 * S_[0][x_a][x_b+2]) * S_[1][y_a][y_b] * S_[2][z_a][z_b]
                                        S[1][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * (y_b * (y_b - 1) * S_[1][y_a][y_b-2] - 2.0 * exp_b * y_b * S_[1][y_a][y_b] - 2.0 * exp_b * (y_b + 1) * S_[1][y_a][y_b] + 4.0 * exp_b**2 * S_[1][y_a][y_b+2]) * S_[2][z_a][z_b]
                                        S[2][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * S_[1][y_a][y_b] * (z_b * (z_b - 1) * S_[2][z_a][z_b-2] - 2.0 * exp_b * z_b * S_[2][z_a][z_b] - 2.0 * exp_b * (z_b + 1) * S_[2][z_a][z_b] + 4.0 * exp_b**2 * S_[2][z_a][z_b+2])
                                    if integral_type == 'angular momentum':
                                        S[0][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[0][x_a][x_b] * ((S_[1][y_a+1][y_b] + (A[1] - C[1]) * S_[1][y_a][y_b]) * (z_b * S_[2][z_a][z_b-1] - 2.0 * exp_b * S_[2][z_a][z_b+1]) - (S_[2][z_a+1][z_b] + (A[2] - C[2]) * S_[2][z_a][z_b]) * (y_b * S_[1][y_a][y_b-1] - 2.0 * exp_b * S_[1][y_a][y_b+1]))
                                        S[1][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[1][y_a][y_b] * ((S_[2][z_a+1][z_b] + (A[2] - C[2]) * S_[2][z_a][z_b]) * (x_b * S_[0][x_a][x_b-1] - 2.0 * exp_b * S_[0][x_a][x_b+1]) - (S_[0][x_a+1][x_b] + (A[0] - C[0]) * S_[0][x_a][x_b]) * (z_b * S_[2][z_a][z_b-1] - 2.0 * exp_b * S_[2][z_a][z_b+1]))
                                        S[2][idx_a + count_a][idx_b + count_b] += coeff_a * coeff_b * S_[2][z_a][z_b] * ((S_[0][x_a+1][x_b] + (A[0] - C[0]) * S_[0][x_a][x_b]) * (y_b * S_[1][y_a][y_b-1] - 2.0 * exp_b * S_[1][y_a][y_b+1]) - (S_[1][y_a+1][y_b] + (A[1] - C[1]) * S_[1][y_a][y_b]) * (x_b * S_[0][x_a][x_b-1] - 2.0 * exp_b * S_[0][x_a][x_b+1]))
                                    count_b += 1
                            count_a += 1
    if integral_type == 'dipole' or integral_type == 'angular momentum':
        S *= -1.0
    if integral_type == 'kinetic':
        S = -0.5 * oe.contract('abc->bc', S)

    return S

    # Note: The recurrence should target an angular momentum component within that particular primative.
    # So for every primative, you will calculate S_00 then from there calculate the angular momentum of
    # the shell you are in.









































