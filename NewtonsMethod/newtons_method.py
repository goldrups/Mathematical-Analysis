# newtons_method.py
"""Volume 1: Newton's Method.
Samuel Alvin Goldrup
Math 347
23 January 2022
"""

import xdrlib
import numpy as np
from autograd import grad
from scipy import optimize
from autograd import numpy as anp
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    #must be careful no bugs when you do the n=1 be very rigorous
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    converged = False
    num_iter = maxiter #will get changed to the iteration no. once converged==True
    if np.isscalar(x0) == True:
        for k in range(maxiter):
            x1 = x0 - alpha*(f(x0) / Df(x0))
            if np.abs(x1-x0) < tol:
                converged = True
                num_iter = k + 1 #because k started from zero
                break
            x0 = x1
    else: #finite dim
        for k in range(maxiter):
            #print(k)
            y = la.solve(Df(x0),f(x0)) #get the "quotient"
            x1 = x0 - alpha*y #iterate
            if la.norm(x1-x0,ord=6) < tol:
                converged = True
                num_iter = k + 1
                break
            x0 = x1 #reassign
    
    return x1, converged, num_iter


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    r = sy.symbols('r') #get the r variable as a symbol
    f = P1*((1+r)**N1-1) - P2*(1-(1+r)**(-N2)) #f(r)
    df = sy.diff(f,r) #get derivative
    df = sy.lambdify(r,df,"numpy") #lambdify f and df
    f = sy.lambdify(r,f,"numpy")
    r = newton(f,0.1,df)[0] #run Newton's method
    return r


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    alpha_space = np.linspace(0.01,0.99,99)
    num_iters = [newton(f,x0,Df,tol,maxiter,alpha)[2] for alpha in alpha_space] #iterations for each alpha
    mindex = np.argmin(num_iters) #get the index of the minimizing alpha
    opt_alpha = alpha_space[mindex]

    plt.plot(alpha_space, num_iters)
    plt.xlabel("alpha values in backtracking")
    plt.ylabel("number of iterations")
    plt.title("finding an optimal alpha")
    plt.show()

    return opt_alpha


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    f = lambda x: np.array([5*x[0]*x[1] - x[0]*(1 + x[1]),-x[0]*x[1] + (1-x[1])*(1+x[1])]) #function 
    df = lambda x: np.array([[5*x[1] - (1+x[1]),5*x[0]-x[0]],[-x[1],-x[0] - 2*x[1]]]) #jacobian

    #first for alpha=1
    for j in np.linspace(-0.25,0,50): #y values
        for k in np.linspace(0,0.25,50): #x values
            guess_pt = np.array([j,k]) #initial point guess
            first = newton(f,guess_pt,df)[0] #one of the points
            if np.allclose(np.abs(first),np.array([0,1])):
                second = newton(f,guess_pt,df,alpha=0.55)[0] #another one of the points
                if np.allclose(second, np.array([3.75,0.25])): #joint convergence
                    return guess_pt


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    x_real = np.linspace(domain[0], domain[1], res) # Real parts.
    x_imag = np.linspace(domain[2], domain[3], res) # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag

    for i in range(iters): #run newton's method iters times
        X_1 = X_0 - f(X_0)/Df(X_0)
        X_0 = X_1
    
    X_K = X_0 #save final result

    Y = np.zeros((res,res))
    
    for j in range(res):
        for k in range(res):
            Y[j,k] = np.argmin(np.abs(X_K[j,k]-zeros)) #find the corresponding zero

    plt.pcolormesh(x_real, x_imag, Y, cmap="brg")
    plt.xlabel("reals")
    plt.ylabel("imaginary")
    plt.title("Basins of attraction")
    plt.show()

if __name__ == "__main__":
    # #f = lambda x: np.exp(x) - 2
    # #df = grad(f)
    # #x0 = anp.array(0.650)
    # #newton(f,x0,df)
    # g = lambda x: x**4 - 3
    # dg = grad(g)
    # print(newton(g,0.8,dg,tol=1e-9,maxiter=50)) #not working with integer input
    # print(optimize.newton(g,0.8,dg, tol=1e-9,full_output=True))

    # print("r=",prob2(30,20,2000,8000))

    # f = lambda x: np.sign(x) * np.power(np.abs(x),1./3)
    # df = grad(f)
    
    # print("alpha=1:",newton(f,0.01,df,alpha=1))
    # print("alpha=0.4:",newton(f,0.01,df,alpha=0.4))

    # f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    # df = lambda x: np.sign(x) * (1/3) * np.power(np.abs(x), -2./3)

    # optimal_alpha(f,0.01,df,maxiter=100)

    # f = lambda x: np.array([np.sin(x[0])*x[1],x[0]*x[1]])
    # df = lambda x: np.array([[x[1]*np.cos(x[0]),np.sin(x[0])],[x[1],x[0]]])

    # x0 = np.array([0.7,0.7])
    # print(newton(f,x0,df))

    # print(optimal_alpha(f,x0,df)

    # f = lambda x: x**3 - 1
    # df = lambda x: 3*x**2
    # zeros_f = np.array([1,-0.5+0.866025404j,-0.5-0.866025404j])

    # g = lambda x: x**3 - x
    # dg = lambda x: 3*x**2 - 1
    # zeros_g = np.array([-1,0,1])
    
    # domain = np.array([-1.5,1.5,-1.5,1.5])

    # plot_basins(f, df, zeros_f, domain, res=1000, iters=15)
    # plot_basins(g, dg, zeros_g, domain, res=1000, iters=15)
    pass