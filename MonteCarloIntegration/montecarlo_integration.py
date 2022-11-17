# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
Samuel Goldrup
MATH 347
14 February 2022
"""

import numpy as np
import scipy.stats
from scipy.stats import uniform
from matplotlib import pyplot as plt


# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    sample = uniform.rvs(size=(n,N))**2 #sample from the uniform distribution

    #truth_arr = sample.sum(axis=0) <= 1 #see which points are in the open ball
    ratio = np.mean(sample.sum(axis=0) <= 1) #take the average
    vol = (ratio*2**n)
    # sr = [(val-ratio)**2 for val in truth_arr]
    # var = np.sum(sr) / (N-1)
    # se = np.sqrt(var/N)
    # vol_se = (se)
    return vol


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    sample = np.random.uniform(a,b,N) #generate randomly sampled values
    return np.abs(b-a)*(np.mean(f(sample)))


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    mins, maxs = np.array(mins), np.array(maxs)
    num_dims = len(mins)

    pts = np.random.random((N, num_dims)) #N points in R^n where n is num_dims

    #print(pts[:4])

    for i in range(num_dims):
        pts[:,i] *= (maxs[i] - mins[i]) 
        pts[:,i] += mins[i]

    #print(pts[:4])

    measure = np.product(maxs-mins) #get the measure

    #print("pts", pts[:10])
    f_vals = [f(pts[j]) for j in range(N)]
    #print(f_vals[:10], len(f_vals))

    return measure * np.mean(f_vals)


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    n = 4 
    mins, maxs = [-3/2,0,0,0], [3/4,1,1/2,1] #initialize mins and maxes
    f = lambda x: 1/((2*np.pi)**(n/2)) * np.exp(-0.5*np.sum([x[i]**2 for i in range(n)])) #put inner product into the function

    means, cov = np.zeros(n), np.eye(n) #start generating the "perfect" result
    perfect = scipy.stats.mvn.mvnun(mins, maxs, means, cov)[0] 

    N_vals = np.logspace(1,5,20).astype('int')

    rel_err = []

    for N in N_vals:
        mc_int = mc_integrate(f,mins,maxs,N)
        rel_err.append(np.abs(perfect-mc_int) / np.abs(perfect)) #relative error calculation

    plt.loglog(N_vals, rel_err,marker=".",label="Relative Error")
    plt.loglog(N_vals,1/np.sqrt(N_vals),marker=".",label="1/sqrt(N)")
    plt.legend()

    plt.show()

# if __name__ == "__main__":
#     print("problem 1:", ball_volume(2, N=100),ball_volume(2, N=1000),ball_volume(2, N=10000),ball_volume(2, N=100000))
#     f = lambda x: x**2
#     g = lambda x: np.sin(x)
#     h = lambda x: 1/x
#     i = lambda x: np.abs(np.sin(10*x)*np.cos(10*x) + np.sqrt(x)*np.sin(3*x))
#     N = 10**7
#     print("problem 2: ", mc_integrate1d(f,-4,2,N),mc_integrate1d(g,-2*np.pi,2*np.pi,N),mc_integrate1d(h,1,10,N),mc_integrate1d(i,1,5,N))


#     j = lambda x: x[0]**2 + x[1]**2
#     j_mins, j_maxs = [0,0], [1,1] 
#     k = lambda x: 3*x[0] - 4*x[1] + x[1]**2
#     k_mins, k_maxs = [1,-2], [3,1]
#     l = lambda x: x[1] + x[2] - x[0]*x[3]**2
#     l_mins, l_maxs = [-1,-2,-3,-4], [1,2,3,4]
#     print("problem 3: ", mc_integrate(j,j_mins,j_maxs), mc_integrate(k,k_mins,k_maxs), mc_integrate(l,l_mins,l_maxs,N=10**4))

#     prob4()
