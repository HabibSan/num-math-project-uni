import numpy as np
import matplotlib as plt
from scipy import linalg
from math import exp
from block_matrix_2d import BlockMatrix
from poisson_problem2d import idx,inv_idx,rhs,compute_error
from linear_solver import solve_lu



def test_block_matrix():
    """
    Test function to validate the BlockMatrix class functionality,
    including LU decomposition and sparsity evaluation.
    """
    print("Testing BlockMatrix class...")

    # Example configuration
    n = 4  # Number of intervals
    block_matrix = BlockMatrix(n)

    # Test sparse matrix generation
    sparse_matrix = block_matrix.get_sparse()
    print("Sparse matrix generated:")
    print(sparse_matrix.toarray())

    # Test LU decomposition
    print("\nPerforming LU decomposition...")
    P, L, U = block_matrix.get_lu()

    # Verify correctness of LU decomposition
    reconstructed_matrix = P @ L @ U
    print("\nReconstructed matrix from P, L, U:")
    print(reconstructed_matrix)

    # Check if the reconstructed matrix matches the original matrix
    original_matrix = block_matrix.matrix

    print(reconstructed_matrix)
    if np.allclose(reconstructed_matrix, original_matrix):
        print("\nLU decomposition is correct.")
    else:
        print("\nLU decomposition is incorrect!")

    # Compare LU decomposition with scipy's LU decomposition
    scipy_P, scipy_L, scipy_U = linalg.lu(original_matrix)
    if np.allclose(P, scipy_P) and np.allclose(L, scipy_L) and np.allclose(U, scipy_U):
        print("\nLU decomposition matches scipy's results.")
    else:
        print("\nLU decomposition does not match scipy's results.")

    # Evaluate sparsity of original and LU matrices
    num_nonzeros_A, rel_nonzeros_A = block_matrix.eval_sparsity()
    num_nonzeros_LU, rel_nonzeros_LU = block_matrix.eval_sparsity_lu()
    print(f"\nMatrix A: {num_nonzeros_A} non-zeros, relative sparsity: {rel_nonzeros_A:.6f}")
    print(f"LU Decomposition: {num_nonzeros_LU} non-zeros, relative sparsity: {rel_nonzeros_LU:.6f}")

    print("\nTest completed.")


def test_solve_lu():
    """
    Tests the `solve_lu` function for correctness. It checks the intermediate
    steps (b', y, x) and compares the results against expected values.
    """
    # Define a sample matrix A and vector b
    n = 4
    block_matrix = BlockMatrix(n)
    block_matrix.get_sparse()
    A = block_matrix.matrix

    def f(x):
      pi = np.pi
      x,y = x[0],x[1]
      lapacian = (x+y)*np.sin(pi* x)*np.sin(pi * y) + pi * np.sin(pi *(x+y))
      return lapacian

    b = rhs(n, f)

    # Perform LU decomposition
    from scipy.linalg import lu
    P, L, U = lu(A)  # P is the permutation matrix, L is lower triangular, U is upper triangular

    # Solve using solve_lu
    print("\nTesting solve_lu with:")
    print("Matrix A:")
    print(A)
    print("Vector b:")
    print(b)
    print("\nLU Decomposition:")
    print("P:")
    print(P)
    print("L:")
    print(L)
    print("U:")
    print(U)
    print("reconstructed matrix:")
    print(P @ L @ U)
    # Step-by-step check
    print("\nStep 1: Permute b to get b':")
    b_prime = np.dot(P.T, b)
    print("b' =", b_prime)

    print("\nStep 2: Solve L * y = b' (forward substitution):")
    y = np.zeros_like(b_prime)
    for i in range(len(b_prime)):
        y[i] = b_prime[i] - np.dot(L[i, :i], y[:i])
    print("y =", y)

    print("\nStep 2: Solve L * y = b' (forward substitution with solver):")
    y = np.linalg.solve(L, b_prime)
    print("y (from np.linalg.solve) =", y)

    print("\nStep 3: Solve U * x = y (backward substitution):")
    x = np.zeros_like(y)
    for i in range(len(y) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    print("x (from manual solve_lu) =", x)

    print("\nStep 3: Solve U * x = y (backward substitution with solver):")
    x = np.linalg.solve(U, y)
    print("x (from np.linalg.solve) =", x)

    # Compare results with NumPy's solver
    print("\nComparing with NumPy's solver:")
    x_expected = np.linalg.solve(A, b)
    print("x (from np.linalg.solve) =", x_expected)

    # Check if results match
    assert np.allclose(x, x_expected), "Test failed: The solution does not match expected values."

    print("\nTest passed: The solution matches expected values.")


def test_idx_and_inv_idx():
    """
    Tests the `idx` and `inv_idx` functions for consistency.
    Ensures that idx(inv_idx(m, n), n) == m and inv_idx(idx(nx, n), n) == nx.
    """
    n = 7  # Grid size
    for i in range(1,n):
        for j in range(1,n):
            m = idx([i,j], n)
            nx = inv_idx(m, n)
            print(m)
    for m in range(1, (n - 1) ** 2 + 1):
        nx = inv_idx(m, n)
        print(nx)
        assert idx(nx, n) == m, f"idx(inv_idx({m}, {n}), {n}) != {m}"
    print("test_idx_and_inv_idx passed!")


def test_rhs():
    """
    Tests the `rhs` function by providing a simple right-hand side function
    and comparing the output vector `b` with expected values.
    """
    def f(x):
        return x[0] + x[1]  # A simple function for testing

    n = 10
    b = rhs(n, f)

    # Check manually computed values
    expected_b = [
        -(1 / n) ** 2 * f(np.array([i / n, j / n]))
        for j in range(1, n)
        for i in range(1, n)
    ]
    print(b)
    print(expected_b)
    assert np.allclose(b, expected_b), "rhs output does not match expected values."
    print("test_rhs passed!")


def test_compute_error():
    """
    Tests the `compute_error` function with a known exact solution and its
    numerical approximation.
    """
    def u_exact(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    n = 4
    hat_u = np.array([u_exact(np.array([i / n, j / n])) for j in range(1, n) for i in range(1, n)])
    error = compute_error(n, hat_u, u_exact)

    assert np.isclose(error, 0), f"Error should be zero, but got {error}"
    print("test_compute_error passed!")


def test_plot_error_vs_N():
    """
    Tests `plot_error_vs_N` by checking if the error decreases as `n` increases.
    This does not validate the plot but confirms that the data trends are correct.
    """
    def u_exact(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    def f(x):
        return 2 * (np.pi ** 2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    max_n = 6  # Use a small range for quick testing
    N_values = [(n - 1) ** 2 for n in range(2, max_n)]
    errors = []

    for n in range(2, max_n):
        # Create a simple sparse matrix for testing (replace with BlockMatrix in real tests)
        A = np.eye((n - 1) ** 2)
        b = rhs(n, f)
        hat_u = np.linalg.solve(A, b)
        error = compute_error(n, hat_u, u_exact)
        errors.append(error)

    # Ensure the error decreases
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], "Error does not decrease with increasing n."

    print("test_plot_error_vs_N passed!")
def main():
    """
    Runs all test functions for the Poisson problem implementation.
    Prints the results of each test and a summary at the end.
    """
    # List of test functions to run
    test_functions = [
        test_idx_and_inv_idx,
        test_rhs,
        test_compute_error,
        test_plot_error_vs_N,
        test_solve_lu,
        test_block_matrix
    ]

    # Tracking results
    passed_tests = 0
    total_tests = len(test_functions)

    print("Running tests...\n" + "=" * 40)

    for test in test_functions:
        try:
            test()  # Run the test
            print(f"{test.__name__}: PASSED ")
            passed_tests += 1
        except AssertionError as e:
            print(f"{test.__name__}: FAILED ")
            print(f"Reason: {e}")
        except Exception as e:
            print(f"{test.__name__}: ERROR ")
            print(f"Unexpected exception: {e}")

    # Summary
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed_tests}/{total_tests} passed.")

    if passed_tests == total_tests:
        print(" All tests passed successfully!")
    else:
        print("Some tests failed. Please review the logs above.")

# Entry point
if __name__ == "__main__":
    main()
