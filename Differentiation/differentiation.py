# differentiation.py
"""Volume 1: Differentiation.
Samuel Goldrup
Math 347
18 January 2022
"""

import sympy as sy
from matplotlib import pyplot as plt
import numpy as np
from autograd import numpy as anp
from autograd import grad, elementwise_grad
import time


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x') #load in the symbols
    expr = (sy.sin(x) + 1)**sy.sin(sy.cos(x)) 
    d_expr = sy.diff(expr,x)
    d_expr_lambdified = sy.lambdify(x,d_expr,"numpy") #make the derivative compatible for nd arrays
    return d_expr_lambdified


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x+h) - f(x)) / h #formulas from the lab file. I assume f is already lambdified

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x+h) - f(x+2*h)) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x-h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) -4*f(x-h) + f(x-2*h)) / (2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h) - f(x-h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) -f(x+2*h)) / (12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    x = sy.symbols('x') #load in the symbol
    f = (sy.sin(x) + 1)**sy.sin(sy.cos(x)) #sy. for sympy functions 
    f = sy.lambdify(x,f,"numpy")
    df = prob1() #use the "perfect" function

    fdq1_err, fdq2_err, bdq1_err, bdq2_err, cdq2_err, cdq4_err = [],[],[],[],[],[]

    for h in np.logspace(-8,0,9):
        fdq1_err.append(np.abs(fdq1(f,x0,h) - df(x0))) #absolute value of error is the difference
        fdq2_err.append(np.abs(fdq2(f,x0,h) - df(x0)))
        bdq1_err.append(np.abs(bdq1(f,x0,h) - df(x0)))
        bdq2_err.append(np.abs(bdq2(f,x0,h) - df(x0)))
        cdq2_err.append(np.abs(cdq2(f,x0,h) - df(x0)))
        cdq4_err.append(np.abs(cdq4(f,x0,h) - df(x0)))

    plt.loglog(np.logspace(-8,0,9),fdq1_err,label="Order 1 forward",marker=".") #np.logspace is powers of 10 from 10^-8 to 10^0
    plt.loglog(np.logspace(-8,0,9),fdq2_err,label="Order 2 forward",marker=".")
    plt.loglog(np.logspace(-8,0,9),bdq1_err,label="Order 1 backward",marker=".")
    plt.loglog(np.logspace(-8,0,9),bdq2_err,label="Order 2 backward",marker=".")
    plt.loglog(np.logspace(-8,0,9),cdq2_err,label="Order 2 centered",marker=".")
    plt.loglog(np.logspace(-8,0,9),cdq4_err,label="Order 4 centered",marker=".")
    plt.ylabel("Absolute Error")
    plt.xlabel("h")
    plt.legend()

    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    a = 500

    plane_data = np.load("plane.npy")
    alphas, betas = plane_data[:,1], plane_data[:,2]
    alphas, betas = np.deg2rad(alphas), np.deg2rad(betas) #convert to radians
    
    x_pos = (a * np.tan(betas)) / (np.tan(betas) - np.tan(alphas)) #get x and y coordinates
    y_pos = (a * np.tan(betas) * np.tan(alphas)) / (np.tan(betas) - np.tan(alphas))

    v_x = [] #x and y directional velocity lists
    v_y = []

    v_x.append(x_pos[1] - x_pos[0]) #first order forward difference quotient
    v_y.append(y_pos[1] - y_pos[0])

    for i in range(1,7): #second order centered difference quotient
        v_x.append((x_pos[i+1] - x_pos[i-1])/2)
        v_y.append((y_pos[i+1] - y_pos[i-1])/2)

    v_x.append(x_pos[-1] - x_pos[-2]) #first order backward difference quotient
    v_y.append(y_pos[-1] - y_pos[-2])

    v_x = np.array(v_x) #convert back to arrays
    v_y = np.array(v_y)

    speed_approx = np.sqrt(v_x**2 + v_y**2) #array broadcast that formula for speed_approx

    return speed_approx

# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    n, m = np.size(x), np.size(f(x)) #m might not be necessary
    I = np.eye(n) #get the standard basis vectors

    cdq2_list = []
    #build each column of J
    for j in range(n):
        cdq2 = (f(x+h*I[:,j]) - f(x-h*I[:,j])) / (2*h)
        cdq2 = cdq2.reshape(-1,1) #turn into columns
        cdq2_list.append(cdq2)

    J = np.hstack((cdq2_list)) #put the columns together into a Jacobian

    return J


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    chebbies = [1,x,2*x**2 - 1] #first 3 polynomials
    n_curr = 2

    if n == 0: #first cheby poly
        return anp.ones_like(x)
    if n == 1: #second cheby poly
        return x
    while n_curr < n:
        chebbies.append(2*x*chebbies[-1] - chebbies[-2])
        n_curr += 1

    return chebbies[-1] #if input for n was >= 2

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    x = anp.linspace(-1,1,100) #must be anp not np
    d_chebby = elementwise_grad(cheb_poly) #derivative at each point
    for i in range(5):
        plt.plot(x,d_chebby(x,i), label='derivative of T_{}'.format(i)) 
    plt.legend()
    plt.show()


# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    rand_x = anp.random.random(N) #the N random points (only between 0 and 1 but shouldn't matter???)


    f = lambda x: (anp.sin(x) + 1)**anp.sin(anp.cos(x)) #lambdified function for autograd

    sympy_times, sympy_derivs, quotient_times = [], [], [] #create lists of derivatives and times
    quotient_derivs, autograd_times, autograd_derivs = [], [], []

    for x0 in rand_x:
        a = time.time()
        exact_deriv = prob1() #get derivative
        sympy_derivs.append(exact_deriv(x0)) #evaluate it at that point
        sympy_times.append(time.time()-a)

        a = time.time()
        quotient_derivs.append(cdq4(f,x0)) #evalute derivative
        quotient_times.append(time.time()-a)

        a = time.time()
        df = grad(f) #get derivative
        autograd_derivs.append(df(x0)) #evalute it at that point
        autograd_times.append(time.time()-a)

    sympy_times, sympy_derivs = np.array(sympy_times), np.array(sympy_derivs) #convert those lists to arrays
    quotient_times, quotient_derivs = np.array(quotient_times), np.array(quotient_derivs)
    autograd_times, autograd_derivs = np.array(autograd_times), np.array(autograd_derivs)

    sympy_errors = np.array([1e-18]*N) #get errors
    quotient_errors = np.abs(quotient_derivs-sympy_derivs)
    autograd_errors = np.abs(autograd_derivs-sympy_derivs)

    plt.scatter(sympy_times,sympy_errors,label="SymPy" ,alpha=0.5)
    plt.scatter(quotient_times,quotient_errors,label="Difference Quotients",alpha=0.5)
    plt.scatter(autograd_times,autograd_errors,label="Autograd",alpha=0.5)

    plt.xlabel("times")
    plt.ylabel("error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    plt.show()