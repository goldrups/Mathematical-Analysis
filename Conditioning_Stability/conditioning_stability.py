# condition_stability.py
"""Volume 1: Conditioning and Stability.
Samuel Goldrup
MATH 347
1 February 2022
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    _, sing_vals, _ = la.svd(A)
    if np.min(sing_vals) == 0: #infinity if noninvertible
        return np.inf
    return np.max(sing_vals) * (1/np.min(sing_vals))


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs()) #get coeffs
    w_roots = np.roots(np.poly1d(w_coeffs)) #get the roots
    w_roots = np.sort(w_roots)
    cond_nums_abs = []
    cond_nums_rel = []

    re = np.array([])
    im = np.array([])

    N = 100
    for i in range(N):
        perturbation = np.random.normal(loc=1,scale=1e-10,size=21)
        new_coeffs = perturbation * w_coeffs #perturb them
        new_roots = np.roots(np.poly1d(new_coeffs)) #get the new roots
        new_roots = np.sort(new_roots)
        k = la.norm(new_roots-w_roots,np.inf) / la.norm(perturbation,np.inf) #get abs cond number
        cond_nums_abs.append(k) 
        next_cond_rel = k*la.norm(w_coeffs, np.inf) / la.norm(w_roots,np.inf) #get relative
        cond_nums_rel.append(next_cond_rel)
        re = np.concatenate((re,new_roots.real)) #add in values
        im = np.concatenate((im,new_roots.imag))


    plt.scatter(re,im,marker=',',label="perturbed",s=1,color="k")
    plt.scatter(w_roots.real,[0]*len(w_roots),label="original") #plot them
    plt.legend()
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.show()    

    return np.mean(cond_nums_abs),np.mean(cond_nums_rel)


# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals) #length of list of perturbed eigenvalues
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    eigs = la.eigvals(A) #get the eigen values
    real = np.random.normal(0,1e-10,A.shape) #perturb !!
    imags = np.random.normal(0,1e-10,A.shape)
    H = real + imags
    A_H = A+H #perturb that matrix !!
    new_eigs = la.eigvals(A_H)
    new_eigs = reorder_eigvals(eigs,new_eigs) #reorder for norm calculation
    abs_cond = la.norm(eigs-new_eigs,2) / la.norm(H,2) #condition numbers
    rel_cond = abs_cond * la.norm(A,2) /  la.norm(eigs,2)
    return abs_cond, rel_cond


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    x_vals = np.linspace(domain[0], domain[1], res) # Real parts.
    y_vals = np.linspace(domain[2], domain[3], res) # Imaginary parts.
    X,Y= np.meshgrid(x_vals,y_vals)

    R = np.empty((res,res)) #effectively z values in a function from RxC --> R

    for i,x in enumerate(x_vals): #need enumerate to assign i,j components
        for j,y in enumerate(y_vals): 
            A = np.array([[1,x],[y,1]])
            R[i,j]= eig_cond(A)[1]


    plt.pcolormesh(X,Y,R,cmap="gray_r")
    plt.colorbar()
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    data = np.load("stability_data.npy")
    xk,yk = data.T #separate into cols
    A = np.vander(xk,n+1)
    soln = la.inv(A.T @ A) @ A.T @ yk #OLS method
    Q,R = la.qr(A, mode="economic")
    soln_bette = la.solve_triangular(R,Q.T @ yk) #OLS method but with QR decomposed matrices


    domain = np.linspace(np.min(xk),np.max(xk),100) #data domain
    ATA_inv_proj = np.polyval(soln,domain) #projections
    QR_inv_proj = np.polyval(soln_bette,domain)
    plt.scatter(xk,yk,color="m",s=1,label="data")
    plt.plot(domain,ATA_inv_proj,label="ATA inversion")
    plt.plot(domain,QR_inv_proj,label="QR")
    plt.legend()
    plt.show()

    return la.norm(ATA_inv_proj-yk,2), la.norm(QR_inv_proj-yk,2)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    x = sy.symbols('x') #import symbols


    n_vals = np.arange(1,11)*5 
    

    function = lambda n: (x**int(n)) * sy.exp(x-1)
    integral = np.array([float(sy.integrate(function(i), (x,0,1))) for i in n_vals]) #convert result to a float

    other_function = lambda n: ((-1)**int(n))*(sy.subfactorial(int(n)) - (sy.factorial(int(n)) / np.e))
    I_n = np.array([other_function(j) for j in n_vals]) #convert result to a float
    
    rel_fwd_err = np.abs(I_n-integral) / np.abs(integral) #compare the floats to get errors


    plt.semilogy(n_vals,rel_fwd_err) #plot with log y axis scale
    plt.scatter(n_vals, rel_fwd_err)
    plt.xlabel("n")
    plt.ylabel("relative forward error")
    plt.show()
    

if __name__ == "__main__":
    # B = np.random.random((3,3))
    # C = la.qr(B)[0]
    # print(matrix_cond(C))
    #print(prob2())
    # print(eig_cond(B))
    # print(prob4(domain=[-100, 100, -100, 100], res=200))
    # prob5(10)
    # prob5(11)
    # prob5(12)
    # prob5(13)
    # prob5(14)
    #prob6()
    pass