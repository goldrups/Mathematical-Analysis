# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Sam Goldrup
MATH 345
14 September 2021
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    rand_arr = np.random.normal(size=(n,n)) #square matrix of random numbers
    rand_arr_means = np.mean(rand_arr, axis=1) #array of means of each row
    return np.var(rand_arr_means) 

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    #n_arr = np.arange(0,1000,10)
    n_arr = np.linspace(100,1000,10)
    var_arr = np.array([var_of_means(int(x)) for x in n_arr]) #list comprehension to get array of of variances
    
    plt.plot(n_arr, var_arr) 
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100) 
    sine_func = np.sin(x)
    cosine_func = np.cos(x)
    arctan_func = np.arctan(x)
    plt.plot(x, sine_func, 'b-')
    plt.plot(x, cosine_func, 'k-')
    plt.plot(x, arctan_func, 'g-')
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x_1 = np.linspace(-2,0.999,100) #we get close to x=1 but exclude it
    x_2 = np.linspace(1.001,6,100) #this way the curve looks discontinuous 
    f_x_1 = 1/(x_1-1)
    f_x_2 = 1/(x_2-1)
    plt.plot(x_1, f_x_1, "--m", linewidth=4)
    plt.plot(x_2, f_x_2, "--m", linewidth=4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.show()


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    #formatting the plot of subplots
    x = np.linspace(0,2*np.pi,100) 
    n_rows = 2
    n_cols = 2
    xmin = 0 
    xmax = 2*np.pi #x and y bounds for each subplot
    ymin = -2
    ymax = 2

    for i in range(n_rows*n_cols):
    	plt.subplot(n_rows,n_cols,i+1)
    	plt.axis([xmin, xmax, ymin, ymax])
    
    
    plt.subplot(221).plot(x, np.sin(x),"g-")
    plt.title("sine of x")
    plt.subplot(222).plot(x, np.sin(2*x),"r--")
    plt.title("sine of 2x")
    plt.subplot(223).plot(x, 2*np.sin(x),"b--")
    plt.title("2*sine of x")
    plt.subplot(224).plot(x, 2*np.sin(2*x),"m:")
    plt.title("2*sine of 2x")
    
    plt.suptitle("my fave sine functions!!")
    plt.tight_layout() #makes plot look nice
    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    fatals = np.load("FARS.npy")
    plt.subplot(121).plot(fatals[:,1],fatals[:,2],"k,") #plots fatalities on US map
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.axis("equal")
    
    plt.subplot(122).hist(fatals[:,0],bins=24, range=[0,24]) #histogram 
    plt.axis([0, 24, 0, 10000])
    plt.xlabel("hour of day")
    plt.ylabel("deaths")
    
    plt.tight_layout()
    plt.show()


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    #set x and y bounds for subplots
    xmin = -2*np.pi
    xmax = 2*np.pi
    ymin = -2*np.pi
    ymax = 2*np.pi
    x = np.linspace(xmin,xmax,100)
    y = np.linspace(ymin,ymax,100)
    X, Y = np.meshgrid(x,y) #a grid of (x,y) combinations
    Z = (np.sin(X) * np.sin(Y)) / (X*Y) 
    
    plt.subplot(121)
    plt.pcolormesh(X,Y,Z,cmap="coolwarm",shading="auto") #heat map
    plt.colorbar()
    plt.subplot(122)
    plt.contour(X,Y,Z,10,cmap="coolwarm") #contour plot
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()