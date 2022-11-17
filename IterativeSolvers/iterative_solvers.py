# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Sam Goldrup
Math 347
8 March 2022
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse


# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1

    if as_sparse == True:
        A = sparse.csr_matrix(A)

    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    abs_err = []
    D = np.diag(np.diag(A)) #make a diagonal matrix out of this
    D_inv = np.linalg.inv(D) #invert
    x0 = np.zeros(len(b))
    for i in range(maxiter):
        diff = b - A@x0 #error
        x1 = x0 + D_inv@diff #update
        abs_err.append(la.norm(A@x1 - b,ord=np.inf))
        if la.norm(x1-x0,ord=np.inf) < tol: #convergence check
            k = i
            break
        x0 = x1 #reset

    if plot == True:
        plt.semilogy(list(range(len(abs_err))),abs_err) #plot the errors by iteration
        plt.xlabel("Iteration") #labels
        plt.ylabel("Absolute Error of Approximation")
        plt.title("Convergence of Jacobi Method")
        plt.show()
    return x1

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    abs_err = []
    x0 = np.zeros(len(b))
    k = maxiter #for counting iterations while testing the function
    for i in range(maxiter):
        x1 = x0.copy() #make a copy of x0 and work on it
        #apply the algorithm elementwise
        for j in range(len(b)):
            x1[j] = x0[j] + (1/(A[j][j]))*(b[j] - A[j].T @ x1)
        err = la.norm(x1 - x0,ord=np.inf) #error
        abs_err.append(err) #convergence check
        if err < tol:
            k = i
            break
        x0 = x1

    if plot == True:
        plt.semilogy(list(range(len(abs_err))),abs_err) #error against iters
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.title("Convergence of Gauss-Seidel Method")
        plt.show()

    return x1

# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    if not isinstance(A, sparse.csr_matrix): #if its not sparse make it sparse
        A = sparse.csr_matrix(A)

    D = A.diagonal() #the diagonal entries
    n = len(b)

    x0 = np.zeros(n)
    for i in range(maxiter):
        x1 = np.copy(x0)
        for j in range(n):
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            #the desired dot product for scipy sparse matrices
            Ajx = A.data[rowstart:rowend] @ x1[A.indices[rowstart:rowend]]
            x1[j] += ((b[j] - Ajx)/D[j]) #update elementwise
        if la.norm(x1-x0,ord=np.inf) < tol: #convergence check
            break
        x0 = x1
    return x1

# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    if not isinstance(A, sparse.csr_matrix): #if its not sparse make it sparse
        A = sparse.csr_matrix(A)

    D = A.diagonal() 
    #same algorithm as function, except we allow for a "relaxation factor"
    #this replaces the 1 in the numerator above D[j] and it coded in as a "omega"
    n = len(b)
    converged = False
    iters = maxiter
    x0 = np.zeros(n)
    for i in range(maxiter):
        x1 = np.copy(x0)
        for j in range(n):
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Ajx = A.data[rowstart:rowend] @ x1[A.indices[rowstart:rowend]]
            x1[j] += (omega*(b[j] - Ajx)/D[j])
        if la.norm(x1-x0,ord=np.inf) < tol: #convergence check
            converged = True
            iters = i + 1
            break
        x0 = x1
    return x1,converged,iters

# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    diagonals = [1,-4,1]
    offsets = [-1,0,1]
    B = sparse.diags(diagonals, offsets, shape=(n,n)) #build B

    A = sparse.block_diag([B]*n) #n of these B matrices
    A.setdiag(1, -n)
    A.setdiag(1, n)

    A = sparse.csr_matrix(A) #initial hotplace in sparse form
    b_ = np.zeros(n)
    b_[0], b_[-1] = -100, -100
    b = np.tile(b_,n) #repeated entries in the vector

    u,converged,iters = sor(A,b,omega,tol,maxiter) #call above function

    u_resh = u.reshape((n,n)) #put it in matrix form for plotting purposes

    if plot == True:
        plt.pcolormesh(u_resh,cmap="coolwarm")
        plt.show()

    return u,converged,iters


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    iter_counts = []
    n = 20
    omegas = np.linspace(1,1.95,20) #all the values of omega to try
    for w in omegas:
        iter_counts.append(hot_plate(n,w,tol=1e-2,maxiter=1000)[2])

    mindex = np.argmin(iter_counts) #get the optimal omega's index

    plt.plot(omegas,iter_counts) #iterations vs omega
    plt.title("Hot Plate Convergence")
    plt.ylabel("Number of Iterations of Convergence")
    plt.xlabel("Omega")
    plt.show()

    return omegas[mindex] #optimal omega

#test code

def test_gauss_seidel_sparse():
    results = []
    for n in [5, 10, 50, 100]:
        A = diag_dom(n)
        b = np.random.rand(n)
        sgs_sol = gauss_seidel_sparse(A,b,)
        sol = la.solve(A,b)
        results.append(np.allclose(sgs_sol, sol))
    return np.alltrue(results), results

def test_sor():
    results = []
    for n,omega in [(5,1.1), (10,1.2), (50,1.1), (100,1.3)]:
        A = diag_dom(n)
        b = np.random.rand(n)
        sor_sol = sor(A,b,omega)[0]
        sol = la.solve(A,b)
        results.append(np.allclose(sor_sol, sol))
    return np.alltrue(results), results


if __name__ == "__main__":
    # b = np.random.random(5)
    # D = diag_dom(len(b))
    # A = np.random.random((5,5))
    # print(A)
    # print(jacobi(D,b,plot=True))
    # print(gauss_seidel(D,b,plot=True))
    # A = diag_dom(25, as_sparse=True)
    # b = np.random.random(25)
    # print(gauss_seidel_sparse(A, b))
    # print(test_gauss_seidel_sparse())
    # print(test_sor())
    # print(hot_plate(20,omega=1.75,plot=True))
    # print(prob7())
    pass