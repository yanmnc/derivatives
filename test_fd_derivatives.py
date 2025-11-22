'''This module tests the implementation of the 
finite difference derivative solver
'''

import matplotlib.pyplot as plt
import numpy as np

import fd_derivatives

if __name__ == "__main__":

    NARRAY = 10
    points_dir2 = np.logspace(1, 3, num=NARRAY, dtype=int)
    points_dir1 = np.copy(points_dir2)

    rel_errors_first_derdir2 = np.zeros(NARRAY)
    rel_errors_first_derdir1 = np.zeros(NARRAY)
    rel_errors_second_derdir2 = np.zeros(NARRAY)
    rel_errors_second_derdir1 = np.zeros(NARRAY)

    for index, (nbpoints_dir2, nbpoints_dir1) in enumerate(zip(points_dir2, points_dir1)):

        meshdir2, meshdir1 = np.meshgrid(
            np.linspace(0., 1., nbpoints_dir2, endpoint=False),
            np.linspace(0., 1., nbpoints_dir1, endpoint=False)
        )

        STEP_DIR2 = 1. / nbpoints_dir2
        STEP_DIR1 = 1. / nbpoints_dir1

        # function to differentiate
        test_func = np.cos(2*np.pi*meshdir2)*np.sin(2*np.pi*meshdir1)

        # predictions of derivatives
        first_derivative_dir2pred = -2*np.pi * \
            np.sin(2*np.pi*meshdir2)*np.sin(2*np.pi*meshdir1)
        first_derivative_dir1pred = 2*np.pi * \
            np.cos(2*np.pi*meshdir2)*np.cos(2*np.pi*meshdir1)

        second_derivative_dir2pred = -4*np.pi**2 * test_func
        second_derivative_dir1pred = -4*np.pi**2 * test_func

        # derivatives computed using finite difference schemes
        first_derivative_dir2_fd = fd_derivatives.periodic_centered_derivative_dir2(
            test_func, STEP_DIR2)
        first_derivative_dir1_fd = fd_derivatives.periodic_centered_derivative_dir1(
            test_func, STEP_DIR1)

        second_derivative_dir2_fd = fd_derivatives.periodic_centered_second_derivative_dir2(
            test_func, STEP_DIR2)
        second_derivative_dir1_fd = fd_derivatives.periodic_centered_second_derivative_dir1(
            test_func, STEP_DIR1)

        rel_errors_first_derdir2[index] = (
            np.sum((first_derivative_dir2_fd - first_derivative_dir2pred)**2)
            / np.sum(first_derivative_dir2pred**2)
        )

        rel_errors_first_derdir1[index] = (
            np.sum((first_derivative_dir1_fd - first_derivative_dir1pred)**2)
            / np.sum(first_derivative_dir1pred**2)
        )

        rel_errors_second_derdir2[index] = (
            np.sum((second_derivative_dir2_fd - second_derivative_dir2pred)**2)
            / np.sum(second_derivative_dir2pred**2)
        )

        rel_errors_second_derdir1[index] = (
            np.sum((second_derivative_dir1_fd - second_derivative_dir1pred)**2)
            / np.sum(second_derivative_dir1pred**2)
        )

    plt.figure()
    plt.plot(points_dir2, rel_errors_first_derdir2, 'rx', label='error_L2')
    plt.title('first_derdir2_error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N_\mathrm{dir2}$')
    plt.legend()

    plt.figure()
    plt.plot(points_dir1, rel_errors_first_derdir1, 'rx', label='error_L2')
    plt.title('first_derdir1_error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N_\mathrm{dir1}$')
    plt.legend()

    plt.figure()
    plt.plot(points_dir2, rel_errors_second_derdir2, 'rx', label='error_L2')
    plt.title('second_derdir2_error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N_\mathrm{dir2}$')
    plt.legend()

    plt.figure()
    plt.plot(points_dir1, rel_errors_second_derdir1, 'rx', label='error_L2')
    plt.title('second_derdir1_error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N_\mathrm{dir1}$')
    plt.legend()

    plt.show()
