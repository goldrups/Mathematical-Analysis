# profiling.py
"""Python Essentials: Profiling.
Sam Goldrup
Math 347
4 Jan 2022
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from numba import jit
import time
from matplotlib import pyplot as plt

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]

    N = len(data)
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == N - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()] for line in infile.readlines()]

    rows_to_upd8 = [i for i in range(len(data)-1)][::-1] #list of rows to check

    for row_num in rows_to_upd8:
        for i in range(len(data[row_num])):
            l_kid, r_kid = data[row_num + 1][i], data[row_num + 1][i + 1] #get the number of left child, right child
            data[row_num][i] += max(l_kid, r_kid)

    return data[0][0]





# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1 #counting the number of primes for the while loop to eventually terminate
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]
    current = 3
    while len(primes_list) < N:
        n = int(np.sqrt(current))
        isprime = True
        for prime in primes_list:
            if current % prime == 0:
                isprime = False #if divisible by prime then it is composite
                break
            if prime > n:
                break
        if isprime:
            primes_list.append(current)
        current += 2
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]): #do the function below just using a for loop
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm(A-x.reshape(-1,1),axis=0)) #get the column of the smallest norm of difference


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(',')) #use i/o functions to get data from the file
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)): #iterate through whole alphabet
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value #multiply index by score and add it to total
    return total


def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile: #open the file
        names = sorted(infile.read().replace('"', '').split(',')) #use i/o functions to get it into a list

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alpha_score = dict((j,i+1) for i,j in enumerate(list(alphabet)))
    name_scores = [sum([alpha_score[let] for let in name]) for name in names] #list comprehension
    return sum([name_scores[i]*(i+1) for i in range(len(name_scores))])



# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    fib_seq = [1,1] #first two terms are easy
    j = 0
    while True:
        if j < 2: #handle first two terms
            yield fib_seq[j]
            j += 1
        else:
            fib_seq.append(fib_seq[-1] + fib_seq[-2]) #else add the previous two
            yield fib_seq[-1]

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i,x in enumerate(fibonacci()):
        if x >= 10**(N-1):
            return i+1 #add one to get the right index


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    sieve = np.arange(2,N)
    while len(sieve) > 0:
        special = sieve[0]
        sieve = sieve[sieve % sieve[0] != 0] #
        yield special


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit #decorator
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    powers = [2**k for k in range(2,8)] #initialize sizes
    times_mp = [] #different times
    times_mpn = []
    times_mpl = []

    for m in powers:
        A = np.random.random((m,m))

        time_pre_mp = time.time() #time before running function
        matrix_power(A,n)
        time_post_mp = time.time() #time after running function
        times_mp.append(time_post_mp-time_pre_mp) #append difference to list of times

        time_pre_mpn = time.time()
        matrix_power_numba(A,n)
        time_post_mpn = time.time()
        times_mpn.append(time_post_mpn-time_pre_mpn)

        time_pre_mpl = time.time()
        np.linalg.matrix_power(A,n)
        time_post_mpl = time.time()
        times_mpl.append(time_post_mpl-time_pre_mpl)

    plt.loglog(powers,times_mp, 'r', base=2, label="matrix power")
    plt.loglog(powers,times_mpn, 'm', base=2, label="matrix power numba")
    plt.loglog(powers,times_mpl, 'k', base=2, label="numpy")
    plt.legend() #shows the labels of the different plots
    plt.xlabel("log of power")
    plt.ylabel("log of time")
    plt.title("run times of different row sum functions")
    plt.show()
