# linear_systems.py
"""Volume 1: Linear Systems.
Sam Goldrup
MATH 345
12 October 2021
"""

import numpy as np
from random import random
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
import numpy as np
import time
from matplotlib import pyplot as plt

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    num_rows = np.shape(A)[0] #used to be numRows
    num_cols = np.shape(A)[1] #used to be numCols
    for i in range(num_cols):
        j = i + 1
        for k in range(j, num_rows): #used to be numRows
            A[k] = A[k] - (A[i])*((A[k,i])/(A[i,i])) #subtract a scalar multiply of row i from row k
    return A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    U=np.copy(A)
    L=np.identity(len(A))
    m, n = np.shape(A) #number of rows and columns, respectively
    for j in range(n):
        for i in range(j+1, m):
            L[i,j] = U[i,j] / U[j,j] #builds lower triangular matrix
            U[i,:] = U[i,:] - (L[i,j]*U[j,:]) #builds upper triangular matrix
    return L,U



# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """

    L,U = lu(A)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(y,L[i,:])) / L[i,i]

    #back subtitution
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1,-1,-1): #iterates backwards on length of b, operating on rows of U
        x[i] = (y[i] - np.dot(x,U[i,:])) / U[i,i]

    return x


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    num_iters = 12 #number of different matrices to test
    sizes = [2**n for n in range(num_iters)] #creates matrices of size 2,4,8,16,...
    times_la_inv_mult_b = [] #initialize lists of times
    times_la_solve = []
    times_la_lu_fact_and_solve = []
    times_la_lu_solve_only = []
    for size in sizes:
        rand_b = [random() for i in range(size)]
        rand_A = [[random() for j in range(size)] for i in range(size)]

        #time la.inv then left mult
        start_la_inv_mult_b = time.time()
        A_inv = la.inv(rand_A)
        np.matmul(A_inv, rand_b)
        la_inv_mult_b_time = time.time() - start_la_inv_mult_b
        times_la_inv_mult_b.append(la_inv_mult_b_time)

        #time la.solve
        start_la_solve = time.time()
        la.solve(rand_A, rand_b)
        la_solve_time = time.time() - start_la_solve
        times_la_solve.append(la_solve_time)

        #time la.lu_factor and la.lu_solve
        start_la_lu_fact_and_solve = time.time()
        L, P = la.lu_factor(rand_A)
        start_la_lu_solve_only = time.time()
        la.lu_solve((L,P), rand_b)
        la_lu_solve_only_time = time.time() - start_la_lu_solve_only
        times_la_lu_solve_only.append(la_lu_solve_only_time)
        la_lu_fact_and_solve_time = time.time() - start_la_lu_fact_and_solve
        times_la_lu_fact_and_solve.append(la_lu_fact_and_solve_time)

    plt.loglog(sizes, times_la_inv_mult_b, 'b.-', lw=2, label="la.inv then left mult")
    plt.loglog(sizes, times_la_solve, 'g.-', lw=2, label="la.solve")
    plt.loglog(sizes, times_la_lu_fact_and_solve, 'r.-', lw=2, label="la.lu_factor then la.lu_solve")
    plt.loglog(sizes, times_la_lu_solve_only, 'k.-', lw=2, label="just la.lu_solve after doing la.lu_factor")
    plt.legend(loc="upper left")
    plt.ylabel("computation time (logged)")
    plt.xlabel("matrix size (logged)")
    plt.title("comparing python matrix computations")

    plt.show()

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    diagonals = [1,-4,1]
    offsets = [-1,0,1]
    B = sparse.diags(diagonals, offsets, shape=(n,n)) #create sparse matrix

    A = sparse.block_diag([B]*n) #make a giant matrix out of B matrices
    A.setdiag(1, -n)
    A.setdiag(1, n)

    return A

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    num_iters = 12 #number of different matrices to test
    sizes = [n**2 for n in range(2,num_iters)]
    times_meth_1 = [] #initialize lists of times
    times_meth_2 = []

    for size in sizes:
        rand_b = [random() for i in range(size**2)] #make random vectors of size 2,4,8,...
        A_csr = prob5(size)
        A_csr = A_csr.tocsr()

        #time method 1
        st_meth_1 = time.time()
        spla.spsolve(A_csr,rand_b)
        time_meth_1 = time.time() - st_meth_1
        times_meth_1.append(time_meth_1)

        #time method 2
        A_csr = A_csr.toarray()
        st_meth_2 = time.time()
        la.solve(A_csr, rand_b)
        time_meth_2 = time.time() - st_meth_2
        times_meth_2.append(time_meth_2)

    plt.loglog(sizes, times_meth_1, 'b.-', lw=2, label="spla.spsolve()")
    plt.loglog(sizes, times_meth_2, 'g.-', lw=2, label="la.solve()")
    plt.legend(loc="upper left")
    plt.ylabel("computation time (logged)")
    plt.xlabel("matrix size (logged)")
    plt.title("comparing computation times w/ giant matrices")

    plt.show()
