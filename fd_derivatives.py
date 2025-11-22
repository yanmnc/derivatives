'''
This module contains functions to compute derivatives using finite difference schemes
'''

import numpy as np


def periodic_centered_derivative_dir1(quantity, grid_step):
    ''' Computes first derivative with second order
    precision along first dimension of quantity, which 
    is a two dimensional array. The grid step in the 
    direction of the derivative is given by step

    The last point of quantity should not be equal to the 
    first point (periodic point not included), i.e. in one 
    dimension quantity[0] != quantity[-1]
    '''
    derivative = np.zeros_like(quantity)
    derivative[1:-1, :] = quantity[2:, :] - quantity[:-2, :]

    # periodic boundary conditions
    derivative[0, :] = quantity[1, :] - quantity[-1, :]
    derivative[-1, :] = quantity[0, :] - quantity[-2, :]

    return derivative / (2 * grid_step)


def periodic_centered_derivative_dir2(quantity, grid_step):
    ''' Computes first derivative with second order
    precision along second dimension of quantity, which 
    is a two dimensional array. The grid step in the 
    direction of the derivative is given by step

    The last point of quantity should not be equal to the 
    first point (periodic point not included), i.e. in one 
    dimension quantity[0] != quantity[-1]
    '''
    return np.transpose(
        periodic_centered_derivative_dir1(np.transpose(quantity),
                                             grid_step))


def periodic_centered_second_derivative_dir1(quantity, grid_step):
    ''' Computes second derivative with second order
    precision along first dimension of quantity, which 
    is a two dimensional array. The grid step in the 
    direction of the derivative is given by step

    The last point of quantity should not be equal to the 
    first point (periodic point not included), i.e. in one 
    dimension quantity[0] != quantity[-1]
    '''
    derivative = np.zeros_like(quantity)
    derivative[1:-1, :] = (
        quantity[2:, :]
        - 2*quantity[1:-1, :]
        + quantity[:-2, :]
    )

    # periodic boundary conditions
    derivative[0, :] = (
        quantity[1, :]
        - 2*quantity[0, :]
        + quantity[-1, :]
    )

    derivative[-1, :] = (
        quantity[0, :]
        - 2*quantity[-1, :]
        + quantity[-2, :]
    )
    return derivative / (grid_step**2)


def periodic_centered_second_derivative_dir2(quantity, grid_step):
    ''' Computes second derivative with second order
    precision along first dimension of quantity, which 
    is a two dimensional array. The grid step in the 
    direction of the derivative is given by step

    The last point of quantity should not be equal to the 
    first point (periodic point not included), i.e. in one 
    dimension quantity[0] != quantity[-1]
    '''
    return np.transpose(
        periodic_centered_second_derivative_dir1(np.transpose(quantity),
                                                    grid_step))
