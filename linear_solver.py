"""
This module consists of only one function solve_lu.
This method is used to compute the LU decomposition of the 
block matrix A coming from the poisson equation.
"""
import numpy as np

def solve_lu(p, l, u, b):
    """
    Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = P * L * U.

    Parameters
    ----------
    p : numpy.ndarray
        permutation matrix of LU-decomposition
    l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
        upper triangular matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """

    #Apply permutation to b to get b'
    b_prime = np.dot(p.T, b)

    #Forward substitution to solve L * y = b'
    y = np.zeros_like(b_prime)
    for i in range(len(b_prime)):
        # y_i = b'_i - sum_{j=1}^{i-1} l_ij * y_j
        y[i] = b_prime[i] - np.dot(l[i, :i], y[:i])

    #Backward substitution to solve U * x = y
    x = np.zeros_like(y)
    for i in range(len(y)-1, -1, -1):
        # x_i = (y_i - sum_{j=i+1}^{n} u_ij * x_j ) / u_ii
        x[i] = (y[i] - np.dot(u[i, i+1:], x[i+1:])) / u[i, i]

    return x
