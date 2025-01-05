"""
This module is intended for constructing and analyzing block matrices coming
from finite difference approximations of the Laplace operator. It
includes functionalities like sparse matrix representation, LU decomposition, sparsity evaluation,
and visualization of non-zero entries and sparsity.

Class
-------
BlockMatrix : Represents block matrices with methods for sparse storage, LU
              decomposition, sparsity evaluation, and condition number computation.


"""
import numpy as np
from scipy.sparse import diags
from scipy.linalg import lu
import matplotlib.pyplot as plt

class BlockMatrix:
    """Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    gridsize: int
        Total number of interior grid points

    Raises
    ------
    ValueError
        If n < 2.
    """

    def __init__(self, n):
        if n < 2:
            raise ValueError("n must be at least 2.")
        self.n = n
        self.gridsize = (n - 1) ** 2  # Total number of interior grid points
        self.matrix = None

    def get_sparse(self):
        """Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """
        # Main diagonal
        main_diag = 4 * np.ones(self.gridsize)

        # Off-diagonals (horizontal neighbors)
        off_diag = -1 * np.ones(self.gridsize - 1)
        # Set zero entries at block boundaries
        for i in range(1, self.n - 1):
            off_diag[i * (self.n - 1) - 1] = 0

        # Side diagonals (vertical neighbors)
        side_diag = -1 * np.ones(self.gridsize - (self.n - 1))

        # Fix: Use unique offsets to avoid ValueError
        offsets = [0, 1, -1, self.n - 1, -(self.n - 1)]

        # If n=2, the offsets will be [0,1,-1,1,-1], which are not unique
        # thats why we proceed by case distinction
        if self.n == 2:
            offsets = [0, 1, -1]
            a_matrix = diags([main_diag, off_diag, off_diag],
                       offsets,
                       shape=(self.gridsize, self.gridsize), format='csr')
        else:
            a_matrix = diags([main_diag, off_diag, off_diag, side_diag, side_diag],
                       offsets,
                       shape=(self.gridsize, self.gridsize), format='csr')

        self.matrix = a_matrix.toarray()  # Store as dense for LU decomposition
        return a_matrix

    def eval_sparsity(self):
        """Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        self.get_sparse()
        num_nonzeros = (self.n-1) * (5*self.n-7) -2
        total_num_entries = self.gridsize**2
        relative_sparsity = num_nonzeros / total_num_entries
        return num_nonzeros, relative_sparsity

    def get_lu(self):
        """Performs LU-Decomposition of the represented matrix A.

        Returns
        -------
        p_matrix : numpy.ndarray
            Permutation matrix of LU decomposition.
        l_matrix : numpy.ndarray
            Lower triangular unit diagonal matrix of LU decomposition.
        u_matrix : numpy.ndarray
            Upper triangular matrix of LU decomposition.
        """
        if self.matrix is None:
            self.get_sparse()  # Ensure matrix is initialized
        p_matrix, l_matrix, u_matrix = lu(self.matrix)

        return p_matrix, l_matrix, u_matrix

    def eval_sparsity_lu(self):
        """Returns the absolute and relative numbers of non-zero elements in
        the LU decomposition.

        Returns
        -------
        int
            Number of non-zeros in the LU decomposition.
        float
            Relative number of non-zeros in the LU decomposition.
        """
        _, l_matrix, u_matrix = self.get_lu()

        # Calculate non-zeros excluding the main diagonal ones of l_matrix
        non_zeros_l = np.count_nonzero(l_matrix) - self.gridsize  # Exclude diagonal ones
        non_zeros_u = np.count_nonzero(u_matrix)
        total_non_zeros_lu = non_zeros_l + non_zeros_u

        total_entries = self.gridsize ** 2
        relative_non_zeros_lu = total_non_zeros_lu / total_entries

        return total_non_zeros_lu, relative_non_zeros_lu

    def get_cond(self):
        """Computes the condition number of the represented matrix.

        Returns
        -------
        float
            Condition number with respect to the infinity-norm.
        """
        return np.linalg.cond(self.matrix, np.inf)

    def plot_nonzeros_sparse_dense_vs_gridsize(self, max_n):
        """Plots the number of non-zero entries of the matrix A 
        as a function of N (total grid points),
        compares sparse and dense storage, and includes reference lines for comparison.

        Parameters
        ----------
        max_n : int
            Maximum value of n to consider for the plot.
        """
        gridsize_values = []
        nonzeros_sparse = []
        nonzeros_dense = []

        # Loop over n values from 2 up to max_n
        for n in range(2, max_n + 1):
            self.n = n
            self.gridsize = (n - 1) ** 2  # Update total grid points
            self.get_sparse()  # Initialize matrix A

            # Store N (number of grid points)
            gridsize_values.append(self.gridsize)

            # Calculate non-zero entries in sparse matrix
            num_nonzeros_sparse, _ = self.eval_sparsity()
            nonzeros_sparse.append(num_nonzeros_sparse)

            # Calculate non-zero entries in dense matrix
            num_nonzeros_dense = self.gridsize ** 2  # For dense matrix, all entries are non-zero
            nonzeros_dense.append(num_nonzeros_dense)

        # Plot results
        plt.figure(figsize=(10, 6))

        # Plot non-zero entries in sparse matrix
        plt.plot(gridsize_values, nonzeros_sparse, label="Sparse A",
                  marker="o", markersize=4, color="blue")

        # Plot non-zero entries in dense matrix
        plt.plot(gridsize_values, nonzeros_dense, label="Dense A",
                  marker="x", markersize=4, color="red")

        # Add reference lines for sparse and dense matrix growth
        plt.plot(gridsize_values, gridsize_values, label=r"$y=N$",
                  linestyle="--", color="lightskyblue")
        plt.plot(gridsize_values, [N**2 for N in gridsize_values], label=r"$y=N^2$",
                  linestyle="--", color="orange")

        # Set axis labels
        plt.xlabel("N", fontsize=12)
        plt.ylabel("Anzahl von Matrix-Einträgen", fontsize=12)

        # Set logarithmic scale for better visualization of large ranges
        plt.xscale("log")
        plt.yscale("log")

        # Customize ticks to display numbers as powers of 10 with LaTeX formatting
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$'))
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$'))

        # Title and grid
        plt.title("Anzahl der Einträge in Sparse und Dense Matrizen vs. N", fontsize=14)
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_nonzeros_matrix_lu_vs_gridsize(self, max_n):
        """Plots the number of non-zero entries of the matrix A and its LU
        decomposition as a function of N (total grid points), with reference lines
        for comparison.

        Parameters
        ----------
        max_n : int
            Maximum value of n to consider for the plot.
        """
        gridsize_values = []
        nonzeros_a = []
        nonzeros_lu = []

        # Loop over n values from 2 up to max_n
        for n in range(2, max_n + 1):
            self.n = n
            self.gridsize = (n - 1) ** 2  # Update total grid points
            self.get_sparse()  # Initialize matrix A

            # Store N (number of grid points)
            gridsize_values.append(self.gridsize)

            # Calculate non-zero entries in A
            num_nonzeros_a, _ = self.eval_sparsity()
            nonzeros_a.append(num_nonzeros_a)

            # Calculate non-zero entries in LU decomposition
            num_nonzeros_lu, _ = self.eval_sparsity_lu()
            nonzeros_lu.append(num_nonzeros_lu)

        # Plot results
        plt.figure(figsize=(10, 6))

        # Plot non-zero entries in A and LU with smaller markers
        plt.plot(gridsize_values, nonzeros_a, label="Nicht-Null Einträge in A",
                  marker="o", markersize=3, color="blue")
        plt.plot(gridsize_values, nonzeros_lu, label="Nicht-Null Einträge in LU",
                  marker="x", markersize=3, color="red")

        # Add reference line for A sparsity
        plt.plot(gridsize_values, gridsize_values,
                  label=r"$y=N$", linestyle="--", color="lightskyblue")

        # Add reference line for LU sparsity
        plt.plot(gridsize_values, [N**1.5 for N in gridsize_values], label=r"$y=N^{1.5}$",
                  linestyle="--", color="lightcoral" )

        # Set axis labels
        plt.xlabel("N", fontsize=12)
        plt.ylabel("Anzahl von Nicht-null Einträgen", fontsize=12)

        # Set logarithmic scale for better visualization of large ranges
        plt.xscale("log")
        plt.yscale("log")

        # Customize ticks to display numbers as powers of 10 with LaTeX formatting
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$'))
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$'))

        # Title and grid
        plt.title("Nicht-null Einträge in Matrix A and LU Zerlegung vs. N", fontsize=14)
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_condition_number_vs_gridsize(self, max_n):
        """Plots the condition number of the matrix A 
        and its LU decomposition matrices (L and U) as functions of N (total grid points).

        Parameters
        ----------
        max_n : int
            Maximum value of n to consider for the plot.
        """
        gridsize_values = []
        cond_numbers_a = []
        cond_numbers_l = []
        cond_numbers_u = []

        for n in range(2, max_n + 1):
            self.n = n
            self.gridsize = (n - 1) ** 2  # Update total grid points
            self.get_sparse()  # Initialize matrix A

            # Store N (number of grid points)
            gridsize_values.append(self.gridsize)

            # Calculate condition number of matrix A
            cond_number_a = self.get_cond()
            cond_numbers_a.append(cond_number_a)

            # Calculate condition numbers for L and U
            _, l_matrix, u_matrix = self.get_lu()
            cond_number_l = np.linalg.cond(l_matrix, np.inf)
            cond_number_u = np.linalg.cond(u_matrix, np.inf)

            cond_numbers_l.append(cond_number_l)
            cond_numbers_u.append(cond_number_u)

        # Plot results
        plt.figure(figsize=(10, 6))

        # Plot condition number of A
        plt.plot(gridsize_values, cond_numbers_a, label=r"cond_{\infty}(A)",
                  marker="o", markersize=4, color="green")

        # Plot condition number of L
        plt.plot(gridsize_values, cond_numbers_l, label=r"cond_{\infty}(L)",
                  marker="x", markersize=4, color="blue")

        # Plot condition number of U
        plt.plot(gridsize_values, cond_numbers_u, label=r"cond_{\infty}(U)",
                  marker="s", markersize=4, color="red")

        # Add reference lines for comparison
        plt.plot(gridsize_values, gridsize_values, label=r"$y=N$",
                  linestyle="--", color="orange")
        plt.plot(gridsize_values, [N**0.5 for N in gridsize_values], label=r"$y=N^{0.5}$",
                  linestyle="--", color="lightgray")

        # Set axis labels
        plt.xlabel("N", fontsize=12)
        plt.ylabel("Kondition", fontsize=12)

        # Set logarithmic scale for better visualization of large ranges
        plt.xscale("log")
        plt.yscale("log")

        # Customize ticks to display numbers as powers of 10 with LaTeX formatting
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$'))
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$'))

        # Title and grid
        plt.title("Kondition von A, L, und U vs. N", fontsize=14)
        plt.legend()
        plt.grid(True)

        plt.show()

def main():
    """This method demonstrates some methods of the blockmatrix class. 
    There are plots to number of entries, nonzero entries and condition numbers vs N
    """
    n = 4
    block_matrix = BlockMatrix(n)
    block_matrix.get_sparse()

    # print A
    print("Sparse Matrix:")
    print(block_matrix.matrix)

    # print absoulute and relative number of nonzero entries in A
    num_nonzeros, relative_sparsity = block_matrix.eval_sparsity()
    print(f"Number of non-zeros: {num_nonzeros}")
    print(f"Relative number of non-zeros: {relative_sparsity}")

    # Calculate and print LU sparsity of A
    total_non_zeros_lu, relative_non_zeros_lu = block_matrix.eval_sparsity_lu()
    print(f"Number of non-zeros in LU decomposition: {total_non_zeros_lu}")
    print(f"Relative number of non-zeros in LU decomposition: {relative_non_zeros_lu}")

    # Plot the number of entries of A as a sparse and dense matrix
    block_matrix.plot_nonzeros_sparse_dense_vs_gridsize(max_n=50)

    # Plot the (number of non-zero entries) sparsity of A and LU for comparison
    block_matrix.plot_nonzeros_matrix_lu_vs_gridsize(max_n=30)

    # Plot the condition of A
    block_matrix.plot_condition_number_vs_gridsize(max_n=40)

if __name__ == "__main__":
    main()
