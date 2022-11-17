# linear_transformations.py
"""Volume 1: Linear Transformations.
Sam Goldrup
345 Section 3
21 September 2021
"""

from random import random
import numpy as np
import time
from matplotlib import pyplot as plt


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    top = np.array([a,0])
    bottom = np.array([0,b])
    mat_trans = np.vstack((top, bottom)) #combines top and bottom rows
    return np.matmul(mat_trans, A)

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    top = np.array([1,a])
    bottom = np.array([b,1])
    mat_trans = np.vstack((top, bottom)) #combines top and bottom rows
    return np.matmul(mat_trans, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    top = np.array([a**2 - b**2, 2*a*b])
    bottom = np.array([2*a*b, b**2 - a**2])
    mat_trans = np.vstack((top, bottom)) #combines top and bottom rows
    scalar = (1/(a**2 + b**2))
    mat_trans = scalar * mat_trans
    return np.matmul(mat_trans, A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    top = np.array([np.cos(theta),-1*np.sin(theta)])
    bottom = np.array([np.sin(theta),np.cos(theta)])
    mat_trans = np.vstack((top, bottom)) #combines top and bottom rows
    return np.matmul(mat_trans, A)


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    num_times = 200 #the number of positions we want to plot
    times = np.linspace(0,T,num_times) #list of 200 different times from 0 to 3pi/2
    angles_e = omega_e * times #convert times to angles
    p_e_start = np.array([x_e, 0]) #initial position in the plane
    earth_pos = np.empty([1,2])
    earth_pos = p_e_start #an array of earth position coordinates, length of num_times after loop finishes
    for i in range(1,num_times):
    	p_e = rotate(p_e_start, angles_e[i]) #calls rotate function
    	earth_pos = np.column_stack((earth_pos,p_e))

    angles_m = omega_m * times #a similar process is done for the moon
    p_m_start = np.array([x_m-x_e, 0]) #initial position relative to earth
    moon_rel_pos = np.empty([1,2])
    moon_rel_pos = p_m_start
    for i in range(1,num_times):
        p_m_rel = rotate(p_m_start, angles_m[i])
        moon_rel_pos = np.column_stack((moon_rel_pos,p_m_rel))


    moon_pos = moon_rel_pos + earth_pos #convert from relative to actual positions

    plt.plot(earth_pos[0], earth_pos[1], 'b-')
    plt.plot(moon_pos[0], moon_pos[1], 'r-')
    plt.gca().set_aspect("equal")
    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the m    #print(earth_pos)
    #print(moon_rel_pos)
    #print(moon_pos)atrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    num_iters = 9
    sizes = [2**x for x in range(num_iters)] #size of n for vectors length n and n*n matrices
    times_m_v = [] #times for matrix vector multiplication
    times_m_m = [] #times for matrix matrix multiplication
    for size in sizes:
    	A = random_matrix(size) #create random matrices and vector
    	B = random_matrix(size)
    	v = random_vector(size)

    	start_t_v = time.time() #time matrix_Vector multiplication for a given size
    	matrix_vector_product(A, v)
    	elapse_t_v = time.time() - start_t_v
    	times_m_v.append(elapse_t_v)

    	start_t_m = time.time() #time matrix_matrix multiplication for a given size
    	matrix_matrix_product(A,B)
    	elapse_t_m = time.time() - start_t_m
    	times_m_m.append(elapse_t_m)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(sizes, times_m_v) #plot matrix vector mult times
    axs[0].scatter(sizes, times_m_v)
    axs[0].set_title("Matrix-Vector Multiplication")
    axs[1].plot(sizes, times_m_m) #plot matrix matrix mult times
    axs[1].scatter(sizes, times_m_m)
    axs[1].set_title("Matrix-Matrix Multiplication")
    plt.tight_layout()
    plt.show()

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    num_iters = 9
    sizes = [2**x for x in range(num_iters)] #size of n for vectors length n and n*n matrices
    times_m_v = [] #times for matrix vector multiplication
    times_m_m = [] #times for matrix matrix multiplication
    times_np_m_v = [] #times for numpy matrix vector multiplication
    times_np_m_m = [] #times for numpy matrix matrix multiplication
    for size in sizes:
        A = random_matrix(size)
        B = random_matrix(size)
        v = random_vector(size)

        start_t_v = time.time() #time matrix_Vector multiplication for a given size
        matrix_vector_product(A, v)
        elapse_t_v = time.time() - start_t_v
        times_m_v.append(elapse_t_v)

        start_t_m = time.time() #time matrix_matrix multiplication for a given size
        matrix_matrix_product(A,B)
        elapse_t_m = time.time() - start_t_m
        times_m_m.append(elapse_t_m)
        start_np_t_v = time.time() #time numpy matrix_vector multiplication for a given size
        np.dot(A,v)
        elapse_np_t_v = time.time() - start_np_t_v
        times_np_m_v.append(elapse_np_t_v)

        start_np_t_m = time.time() #time numpy matrix_matrix multiplication for a given size
        np.dot(A,B)
        elapse_np_t_m = time.time() - start_np_t_m
        times_np_m_m.append(elapse_np_t_m)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(sizes, times_m_v) #plot linear scale
    axs[0].scatter(sizes, times_m_v)
    axs[0].plot(sizes, times_m_m)
    axs[0].scatter(sizes, times_m_m)
    axs[0].plot(sizes, times_np_m_v)
    axs[0].scatter(sizes, times_np_m_v)
    axs[0].plot(sizes, times_np_m_m)
    axs[0].scatter(sizes, times_np_m_m)
    axs[0].set_title("Matrix-Vector/Matrix Multiplication (Linear)")
    axs[1].plot(sizes, times_m_v) #plot log scale
    axs[1].scatter(sizes, times_m_v)
    axs[1].plot(sizes, times_m_m)
    axs[1].scatter(sizes, times_m_m)
    axs[1].plot(sizes, times_np_m_v)
    axs[1].scatter(sizes, times_np_m_v)
    axs[1].plot(sizes, times_np_m_m)
    axs[1].scatter(sizes, times_np_m_m)
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_title("Matrix-Vector/Matrix Multiplication (Logarithmic)")
    plt.tight_layout()
    plt.show()
