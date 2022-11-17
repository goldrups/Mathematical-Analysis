# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Sam Goldrup
MATH 345
19 October 2021
"""
import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A): #this gives negative values
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m, n = np.shape(A) #get dimensions
    Q = A.copy()
    R = np.zeros((n,n))
    for i in range(n): #iterate on columns
        R[i,i] = la.norm(Q[:,i]) 
        Q[:,i] = Q[:,i] / R[i,i] #normalize first column
        for j in range(i+1,n): #iterate on i+1th to n-1th row
            R[i,j] = (Q[:,j]).T @ Q[:,i] #standard inner product of F^n
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q,R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    m,n = np.shape(A) #get dimensions
    Q,R = qr_gram_schmidt(A)
    abs_det_Q = 1 #by defn
    abs_det_R = 1 #because we mult
    print(R)
    print(abs_det_R)
    for i in range(n):
        abs_det_R *= np.abs(R[i,i]) #successive multiplication down the diagonal
        print(abs_det_R)
    abs_det_A = abs_det_R * abs_det_Q
    return abs_det_A


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q, R = qr_gram_schmidt(A) #call the function from first problem
    Q_T = np.transpose(Q) #transpose it
    Q_T_b = np.matmul(Q_T, b) #the matrix vector product

    n=len(Q_T_b)
    x=np.array([0.0 for i in range(n)]) #initialize an array of 0's
    for i in range(n-1,-1,-1): #iterate backwards from n-1th column to 0th column
        r=(Q_T_b[i]-sum([x[j]*R[i][j] for j in range(i+1,n)]))/R[i][i]
        x[i]=r #assign elements of the soln vector
    return x

sign = lambda x: 1 if x >=0 else -1

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A)
    R = A.copy()
    Q = np.identity(m) #we build Q starting from an identity matrix
    for k in range(n):
        u = np.copy(R[k:,k]) #copy part of the matrix 
        u[0] = u[0] + (np.sign(u[0]) * la.norm(u)) #assign first element of the copy
        u = u / la.norm(u) #normalize
        R[k:,k:] = R[k:,k:] - 2 * np.outer(u, (np.transpose(u) @ R[k:,k:])) #pefrom householder operations
        Q[k:,:] = Q[k:,:] - 2 * np.outer(u, (np.transpose(u) @ Q[k:,:]))

    return np.transpose(Q), R




# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m,n = np.shape(A)
    H = A.copy()
    Q = np.identity(m) #we build Q starting from an identity matrix
    for k in range(n-2): #iterate on columns from 0th col to n-1th col
        u = np.copy(H[k+1:,k]) #make a copy
        u[0] = u[0] + (np.sign(u[0]) * la.norm(u)) #assign first element of the copy
        u = u / la.norm(u) # normalize it
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,(np.transpose(u) @ H[k+1:,k:])) #perform hessenberg operations
        H[:,k+1:] = H[:,k+1:] - 2*np.outer((H[:,k+1:] @ u),np.transpose(u))
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,(np.transpose(u) @ Q[k+1:,:]))
    return H, np.transpose(Q)
