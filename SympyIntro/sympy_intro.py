# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
Samuel Alvin Goldrup
Math 347
4 January 2022
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x,y = sy.symbols('x, y')
    return sy.Rational(2,5)*sy.exp(x**2-y)*sy.cosh(x+y) + sy.Rational(3,7)*sy.log(x*y+1) #rational for fraction


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    x,i,j = sy.symbols('x i j') #put the symbols in the namespace
    expr = sy.product(sy.summation(j*(sy.sin(x) + sy.cos(x)), (j,i,5)), (i,1,5)) #double indexing, inner one first
    expr_expanded = sy.expand(expr) #expand it out
    expr_final = sy.trigsimp(expr_expanded) #simplify the trig functions

    return expr_final


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    x,y,n = sy.symbols('x y n') #put the symbols in the namespace
    expr = sy.summation(x**n / sy.factorial(n), (n,0,N)) #euler expansion
    expr = expr.subs(x, -y**2) #plug in neg y sq
    f = sy.lambdify(y, expr, "numpy") #make it a lambda type

    domain = np.linspace(-2,2) 
    output_lambda = f(domain)
    output = np.exp(-domain**2)

    plt.plot(domain,output,label="e to neg y_sq")
    plt.plot(domain,output_lambda,label="series")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x,y,r,theta = sy.symbols('x y r theta')
    expr_num = (x**2 + y**2)**sy.Rational(7,2) + 18*y*(x**5) - 60*(x**3)*(y**3) + 18*x*(y**5) #numerator of part of expression given to me
    expr_denom = (x**2 + y**2)**3 #denominator of part of expression given to me
    expr = 1 - expr_num /expr_denom #full expression

    new_expr = expr.subs({x:r*sy.cos(theta),y:r*sy.sin(theta)}) #substitution
    new_expr = sy.simplify(new_expr) #simplify it
    soln = sy.solve(new_expr,r) #solve for r
    soln_pick = soln[1]
    f = sy.lambdify(theta,soln_pick,"numpy")

    thetas = np.linspace(0,2*np.pi,1000)
    plt.plot(f(thetas)*np.cos(thetas), f(thetas)*np.sin(thetas)) #plot a rose function
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x,y,l = sy.symbols('x y l')
    A = sy.Matrix([[x-y,x,0],[x,x-y,x],[0,x,x-y]]) #create matrix above
    A_I = A - l*sy.eye(3,3) #multiply identity
    poo = sy.det(A_I) #get the determinant
    evals = sy.solve(poo,l) #get solutions for l
    evecs = [(A-eval*sy.eye(3,3)).nullspace() for eval in evals] #get nullspace of A-Ilambda
    return dict(zip(evals, evecs))


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    x = sy.symbols('x')
    domain = np.linspace(-5,5,1000) #make the domain linspace
    f = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100 #make the function
    f_unction = sy.lambdify(x,f,"numpy") #make it a lambda type
    df = sy.diff(f,x) #derivative 
    crits = np.array(sy.solve(df,x)) #solve for roots

    dg = sy.diff(df,x) #get second derivative 
    h = sy.lambdify(x,dg,"numpy")
    sec_ord_cond = h(crits) #evalute second derivative at critical poinds

    mins = crits[sec_ord_cond > 0] #separte maxs from mins
    maxs = crits[sec_ord_cond < 0]

    plt.plot(domain, f_unction(domain))
    plt.scatter(mins,f_unction(mins),label="minima")
    plt.scatter(maxs,f_unction(maxs),label="maximus")
    plt.legend()
    plt.show()


    return set(mins.tolist()), set(maxs.tolist())


# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    x,y,z,rho,phi,theta,r = sy.symbols('x y z rho phi theta r') 
    f = (x**2 + y**2 + z**2)**2
    f = f.subs({x:rho*sy.sin(phi)*sy.cos(theta),y:rho*sy.sin(phi)*sy.sin(theta),z:rho*sy.cos(phi)}) #substitute for spherical coords
    h = sy.Matrix([rho*sy.sin(phi)*sy.cos(theta),rho*sy.sin(phi)*sy.sin(theta),rho*sy.cos(phi)]) #make h function
    J = h.jacobian([rho,theta,phi]) #get jacobian

    
    absdetJ = -1*sy.det(J) #get det of Jacobian
    integral = sy.integrate(f*absdetJ,(rho,0,r),(theta,0,2*sy.pi),(phi,0,sy.pi)) #specify functions and then bounds of trip integral
    integral_func = sy.lambdify(r,integral,"numpy") #lambdify the result of this to make it a function
    radii = np.linspace(0,3,100)
    plt.plot(radii,integral_func(radii)) 
    plt.show()
    return integral_func(2)