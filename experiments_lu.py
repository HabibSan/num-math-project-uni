"""
Module: Interactive Poisson Problem Solver

This script provides an interactive environment for solving the 2D Poisson problem 
using finite differences. Users can specify custom exact solutions and source terms, 
and visualize results, including the error convergence behavior and solution plots. 

It  supports user-defined examples with adjustable parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
from poisson_problem2d import plot_poisson_problem, plot_error_vs_N

# Interactive input from user for poisson problem
def get_interactive_input():
    """Python problem experiment 
    """
    print("Willkommen zur interaktiven Lösung des Poisson-Problems!")
    n = int(input("Geben Sie die Anzahl der Intervalle n für plot_poisson_problem ein (z.B. 10). \
                  Verwenden sie am Anfang kleine Werte zum Test: "))
    max_n = int(input("Geben Sie eine obere Schranke max_n für plot_error_vs_N ein (z.B. 20)."))
    alpha = float(input("Geben Sie ein Wert für alpha ein (z.B. 30): "))
    kappa = float(input("Geben Sie ein Wert für kappa ein (z.B. 1): "))

    print("\nGeben Sie die Funktion u(x[0], x[1]) als Python-Ausdruck ein.\
           Sie dürfen dabei alpha, und kappa verwenden.\
           Dieses u soll die exakte Lösung des Poissonproblems sein. \
          Achten sie das es die Randbedingungen g = 0 auf dem Rand von [0,1]x[0,1] erfüllt")
    print("Beispiel: x[0] * x[1] * np.sin(kappa * np.pi*x[0]) * np.sin(kappa * np.pi*x[1])")
    u_str = input("u(x[0], x[1]) = ")

    print("\nGeben Sie die Funktion f(x[0], x[1]) als Python-Ausdruck ein.\
           Dieses f soll zu ihrem u gewählt sein und die Poisson gleichung erfüllen ")
    print("Beispiel: 2 * kappa * pi * x[0] * np.cos(kappa * pi * x[1]) \
          * np.sin(kappa * pi * x[0]) + \
    2 * kappa * pi * x[1] * np.cos(kappa * pi * x[0]) * np.sin(kappa * pi * x[1]) - \
    2 * kappa**2 * pi**2 * x[0] * x[1] * np.sin(kappa * pi * x[0]) \
          * np.sin(kappa * pi * x[1])")
    f_str = input("f(x[0], x[1]) = ")

    return n, max_n, alpha, kappa, u_str, f_str

# convert strings to python expressions
def create_function_from_string(func_str, additional_vars):
    """ converts strings to python expressions 
    """
    def func(x):
        variables = {"x": x, **additional_vars, "np": np}
        return eval(func_str, variables)
    return func


# Standard example for inspiration
def user_examples():
    """standard examples with interaction
    """
    print("\n--- Beispiele zur Exploration ---")
    max__n = int(input("Geben Sie eine maximale Auflösung \
                      max__n für die Beispiele ein (z.B. 20): "))
    n_ = int(input("Geben Sie eine maximale Auflösung max__n \
                  für die Beispiele ein (z.B. 20): "))
    print("Es folgt nun ein Beispiel für u(x[0],x[1]):\
           x[0] * x[1] * np.sin(kappa * np.pi*x[0]) * np.sin(kappa * np.pi*x[1])")

    kappa_ = int(input("Geben sie ein Wert für kappa ein (z.B.: 1): "))

    def u(x):
        return x[0] * x[1] * np.sin(kappa_ * np.pi*x[0]) * np.sin(kappa_ * np.pi*x[1])

    def f(x):
        pi = np.pi
        x,y = x[0],x[1]
        laplacian =  2 * kappa_ * pi * x * np.cos(kappa_ * pi * y) * np.sin(kappa_ * pi * x) + \
        2 * kappa_ * pi * y * np.cos(kappa_ * pi * x) * np.sin(kappa_ * pi * y) - \
        2 * kappa_**2 * pi**2 * x * y * np.sin(kappa_ * pi * x) * np.sin(kappa_ * pi * y)
        return -laplacian

    plot_error_vs_N(u,f,max_n = max__n)
    plot_poisson_problem(n_, u, f)


    print("Es folgt nun ein Beispiel für u(x, y) = (x^2 + y^2)^2 * (1 - x) * (1 - y).")
    def u_0(x):
        """Exact solution: u(x, y) = (x^2 + y^2)^2 * (1 - x) * (1 - y)."""
        x, y = x[0], x[1]
        return (x**2 + y**2)**2 * (1 - x) * (1 - y) * x * y

    def f_0(x):
        """Source term: f(x, y) = -Laplace(u_exact_poly)."""
        x, y = x[0], x[1]
        laplacian = 2 * x**6 - 2 * x**5 + 54 * x**4 * y**2 - 42 * x**4 * y - 44 * x**3 * y**2 +\
        32 * x**3 * y + 54 * x**2 * y**4 - 44 * x**2 *\
              y**3 - 42 * x * y**4 + 32 * x * y**3 + 2 * y**6 - 2 * y**5

        return -laplacian

    plot_error_vs_N(u=u_0,f=f_0,max_n = max__n)
    plot_poisson_problem(n_, u=u_0, f=f_0)

    print("Es folgt nun ein Beispiel für u(x,y):\
           (1-np.exp(-x * (1 - x)) )* (1-np.exp(-y * (1 - y))) * (1-x)*(1-y)")
    def u_1(x):
        x,y = x[0],x[1]
        return (1-np.exp(-x * (1 - x)) )* (1-np.exp(-y * (1 - y))) * (1-x)*(1-y)

    def f_1(x):
        pi = np.pi
        x,y = x[0],x[1]
        lapacian = 2 * np.exp(x * (x - 1)) * \
            (np.exp(y * (y - 1)) - 1) * (x - 1) * (y - 1) +\
        2 * np.exp(y * (y - 1)) * (np.exp(x * (x - 1)) - 1) * \
            (x - 1) * (y - 1) + 2 * np.exp(x * (x - 1)) \
        * (2 * x - 1) * (np.exp(y * (y - 1)) - 1) * (y - 1) + \
            2 * np.exp(y * (y - 1)) * (2 * y - 1) * \
        (np.exp(x * (x - 1)) - 1) * (x - 1) + \
            np.exp(x * (x - 1)) * (2 * x - 1)**2 * (np.exp(y * (y - 1)) - 1)\
        * (x - 1) * (y - 1) + np.exp(y * (y - 1)) * \
            (2 * y - 1)**2 * (np.exp(x * (x - 1)) - 1) * (x - 1) * (y - 1)
        return -lapacian

    plot_error_vs_N(u=u_1,f=f_1,max_n = max__n)
    plot_poisson_problem(n_, u= u_1 , f=f_1)

    print("Es folgt nun ein Beispiel für u(x, y) = \
          (x*(1-x))^alpha * (y*(1-y))^alpha, with alpha > 1 ")
    alpha_ = int(input("Geben sie ein Wert für alpha > 1 an: "))
    def u_2(x):
        """Exact solution: u(x, y) = (x*(1-x))^alpha_ * (y*(1-y))^alpha_, with alpha_ > 1"""
        x, y = x[0], x[1]
        return (x * (1 - x))**alpha_ * (y * (1 - y))**alpha_

    def f_2(x):
        """Source term: f(x, y) = -Laplace(u_exact_poly)."""
        x, y = x[0], x[1]
        laplacian = alpha_ * (2 * x - 1)**2 * \
            (-x * (x - 1))**(alpha_ - 2) * \
            (-y * (y - 1))**alpha_ * (alpha_ - 1) - \
                2 * alpha_ * (-x * (x - 1))**(alpha_ - 1) * \
                (-y * (y - 1))**alpha_ - 2 * alpha_ *\
                      (-x * (x - 1))**alpha_ * (-y * (y - 1))**(alpha_ - 1) + \
                    alpha_ * (2 * y - 1)**2 * (-x * (x - 1))**alpha_ *\
                          (-y * (y - 1))**(alpha_ - 2) * (alpha_ - 1)
        return -laplacian

    plot_error_vs_N(u=u_2,f=f_2,max_n = max__n)
    plot_poisson_problem(n_, u= u_2, f= f_2)







# functions for demonstration of 5 x 5 grid
def plot_laplace_grid_5x5():
    fig, ax = plt.subplots(figsize=(6, 6))
    points = [(i, j) for i in range(5) for j in range(5)]
    center = (2, 2)

    for (i, j) in points:
        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
        opacity = max(0.05, 1 - distance * 0.4)

        if (i, j) == center:
            ax.plot(i, j, 'o', color="darkred", markersize=12)
            ax.text(i + 0.05, j + 0.05, r"$u_{ij}$", fontsize=14, color="darkred")
        elif (i, j) in [(center[0] - 1, center[1]), (center[0] + 1, center[1]),
                        (center[0], center[1] - 1), (center[0], center[1] + 1)]:
            ax.plot(i, j, 'o', color="lightcoral", markersize=12, alpha=opacity)
            ax.text(i + 0.05, j + 0.05, r"$u_{%d,%d}$" % (i, j),
                    fontsize=14, color="lightcoral", alpha=opacity)
        else:
            ax.plot(i, j, 'o', color="gray", markersize=12, alpha=opacity)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def main():
    """Demonstration of the methods in this module
      through an interactive experiment conducted by the user
    """
    
    # interactive input
    n, max_n, alpha, kappa, u_str, f_str = get_interactive_input()

    # create functions u and f
    u = create_function_from_string(u_str, {"kappa": kappa, "alpha": alpha})
    f = create_function_from_string(f_str, {"kappa": kappa, "alpha": alpha})

    # invoke plot_error_vs_N and plot_poisson_problem
    plot_error_vs_N(u=u, f=f, max_n=max_n)
    plot_poisson_problem(n=n, u=u, f=f)

    #standard examples
    user_examples()

    plot_laplace_grid_5x5()

if __name__ == "__main__":
    main()
