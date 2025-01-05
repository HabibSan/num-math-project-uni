
"""
This modules goal is solving the 2D Poisson problem using finite differences and LU decomposition.

It provides functions for setting up, solving, and visualizing the
 Poisson problem for [0,1] x [0,1]. The discrete Poisson equation comes from viewing the equation
on a uniform grid. Then LU decomposition is used to solve the resulting linear system Ax = b

Key Features:
-------------
**Error Analysis**: Computation of the error between the numerical solution and the exact solution.
**Visualization**: Visualization of the exact and numerical solutions and their error
**RHS Function**: Computes the right-hand side vector based on given `f` of the poisson problem

Author:
-------
- Habib Eser und Lukas Pauli
"""
import numpy as np
import matplotlib.pyplot as plt
from block_matrix_2d import BlockMatrix
from linear_solver import solve_lu

def idx(nx, n):
    """ Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.

    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    # base index is 1, leads to shift by -1 and respectively +1
    return (nx[0]-1) * (n - 1) + 1 + nx[1]-1

def inv_idx(m, n):
    """ Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.

    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    n : int
        Number of intervals in each dimension.

    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    # base index is 1, leads to shift by -1 and respectively +1
    nx = [(m-1) // (n - 1) + 1,(m-1) % (n-1)+1]

    return nx


def rhs(n, f):
    """Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If n < 2.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    # Initialize the right-hand side vector b
    b = np.zeros((n - 1) ** 2)

    # Single loop to iterate over all grid points
    for m in range(1, (n - 1) ** 2 + 1):
        # Get coordinates of the grid point
        coords = inv_idx(m, n)
        x_coord, y_coord = coords[0] / n, coords[1] / n
        b[m - 1] = (1 / n)**2 * f(np.array([x_coord, y_coord]))

    return b


def compute_error(n, hat_u, u):
    """
    Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm in a 2D setting.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    hat_u : numpy.ndarray
        Finite difference approximation of the solution of the Poisson problem,
        ordered according to `idx`.
    u : callable
        Exact solution of the Poisson problem.
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'
        representing a 2D point (x[0], x[1]).

    Returns
    -------
    float
        Maximum absolute error at the discretization points.
    """

    # Initialize an array to store the exact solution at the grid points
    u_exact_vec = np.zeros((n - 1) ** 2)

    # Single loop to iterate over all grid points
    for m in range(1, (n - 1) ** 2 + 1):
        # Get coordinates of the grid point
        coords = inv_idx(m, n)
        x_coord, y_coord = coords[0] / n, coords[1] / n
        u_exact_vec[m-1] = u(np.array([x_coord, y_coord]))

    # Compute the maximum absolute error using the infinity norm
    error_inf_norm = np.max(np.abs(hat_u - u_exact_vec))

    return error_inf_norm

def solve_poisson_problem(n, f):
    """
    Solves the Poisson problem for a given grid size `n`, exact solution `u`,
    and right-hand side function `f`.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    u : callable
        Exact solution of the Poisson problem.
    f : callable
        Right-hand side function in the Poisson problem.

    Returns
    -------
    numpy.ndarray
        Finite difference approximation of the solution of the Poisson problem,
        ordered according to `idx`.
    """
    # Initialize BlockMatrix and obtain sparse matrix
    block_matrix = BlockMatrix(n)
    block_matrix.get_sparse()

    # Right-hand side vector `b`
    b_rhs = rhs(n, f)

    # LU Decomposition and solving A*hat_u = b
    p, l, u = block_matrix.get_lu()
    hat_u = solve_lu(p, l, u, b_rhs)

    return hat_u

def plot_poisson_problem(n, u, f):
    """
    Plots the solution of the Poisson problem for a given grid size `n`,
    exact solution `u`, and right-hand side function `f`.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    u : callable
        Exact solution of the Poisson problem.
    f : callable
        Right-hand side function in the Poisson problem.
    """
    # Solve the Poisson problem
    hat_u = solve_poisson_problem(n, f)

    # Create a meshgrid for plotting
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Evaluate the exact solution at all grid points
    exact_solution = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            exact_solution[i, j] = u([x_mesh[i, j], y_mesh[i, j]])

    # Approximate solution on the interior grid points
    approx_solution = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0 or i == n or j == 0 or j == n:
                approx_solution[i, j] = u([x_mesh[i, j], y_mesh[i, j]])
            else:
                m = idx([i, j], n)
                approx_solution[i, j] = hat_u[m - 1]

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot exact solution with solid lines
    ax.plot_wireframe(x_mesh, y_mesh, exact_solution, color='orange', linestyle='-'
                      , linewidth=2, label="Exakte Lösung")

    # Plot approximate solution with dashed lines
    ax.plot_wireframe(x_mesh, y_mesh, approx_solution, color='blue'
                      , linestyle = (0,(10,3)), linewidth=1.5, label="Approximation")

    # Labels and titles
    ax.set_title(rf"Graphen von $u$ und $\hat{{u}}$ für $n = {n}$", fontsize=16)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_zlabel("$Funktionswert$", fontsize=14)

    # Add legend
    ax.legend(loc='upper right', fontsize=12)

    # Grid and visuals
    ax.view_init(30, 45)  # Adjust view for better 3D perspective
    plt.tight_layout()
    plt.show()

def plot_error_vs_N(u, f, max_n):
    """
    Plots the error of the numerical solution of the Poisson problem
    as a function of grid size N.

    Parameters
    ----------
    u : callable
        Exact solution of the Poisson problem.
    f : callable
        Right-hand side function in the Poisson problem.
    max_n : int
        Maximum grid size (exclusive).
    """
    gridsize_values = [(n-1)**2 for n in range(2, max_n)]
    errors = []

    for n in range(2, max_n):
        hat_u = solve_poisson_problem(n, f)
        # Compute and store error
        error = compute_error(n, hat_u, u)
        errors.append(error)

    # Create plot
    plt.figure(figsize=(8, 6))

    # Plotting error vs. N
    plt.plot(gridsize_values, errors, marker="o", label="Error"
             , color="#1f77b4", markersize=8, linestyle='-', linewidth=2)

    # Add reference lines for expected convergence orders
    ref_x = np.array(gridsize_values , dtype=float) # Convert to floats to handle negative powers
    plt.plot(ref_x, ref_x**(-0.5), label=r"$\mathcal{O}(N^{-0.5})=\mathcal{O}(h^{1})$"
             , linestyle="--", color="orange", linewidth=2)
    plt.plot(ref_x, ref_x**(-1), label=r"$\mathcal{O}(N^{-1}) =\mathcal{O}(h^{2}) $"
             , linestyle="--", color="green", linewidth=2)

    # Logarithmic scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Labeling and Titles
    plt.xlabel(r"Gitter Größe N", fontsize=14)
    plt.ylabel(r"$\| u - \hat{u} \|_{\infty}$", fontsize=14)
    plt.title("Error vs. Gitter Größe N", fontsize=16)

    # Customize ticks and labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add grid for better visibility
    plt.grid(True, which="both", linestyle="--", color="gray", alpha=0.6)

    # Legend
    plt.legend(loc="best", fontsize=12)

    # Tight layout for better spacing
    plt.tight_layout()

    # Show plot
    plt.show()

def main():
    """Demonstration of the methods in this module on specific functions u, f.
    """
    kappa = 1
    def u(x):
        return x[0] * x[1] * np.sin(kappa * np.pi*x[0]) * np.sin(kappa * np.pi*x[1])

    def f(x):
        pi = np.pi
        x,y = x[0],x[1]
        laplacian =  2 * kappa * pi * x * np.cos(kappa * pi * y) * np.sin(kappa * pi * x) + \
        2 * kappa * pi * y * np.cos(kappa * pi * x) * np.sin(kappa * pi * y) - \
        2 * kappa**2 * pi**2 * x * y * np.sin(kappa * pi * x) * np.sin(kappa * pi * y)
        return -laplacian

    plot_error_vs_N(u,f,max_n = 30)
    plot_poisson_problem(10, u, f)

if __name__ == "__main__":
    main()
