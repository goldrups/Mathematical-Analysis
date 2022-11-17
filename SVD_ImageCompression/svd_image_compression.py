# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread
from matplotlib import pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    A_H = A.conj().T #take the hermitian
    AHA = A_H @ A 
    evals, evecs = la.eig(AHA)
    index_order = np.argsort(np.sqrt(evals))[::-1] #indices of eigen values in descending eigen value order
    evals = evals[index_order] #reorder the evals
    evecs = evecs[:,index_order] #reorder the evecs
    sigmas = np.sqrt(evals) #
    r = len([s for s in sigmas if abs(s) > tol]) #number of positive singular vals
    sigmas_1 = sigmas[:r] #reduced rank approximation
    V_1 = evecs[:,:r]
    U_1 = (A @ V_1) / sigmas_1
    return U_1, sigmas_1, V_1.conj().T, 


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    x,y = np.sin(np.linspace(0, 2*np.pi, 200)), np.cos(np.linspace(0, 2*np.pi, 200)) #initialize a buttload of x,y pts
    S = np.vstack((x,y)) #make a matrix out of it
    E = np.array([[1,0,0],[0,0,1]]) #matrix containing basis vectors
    U,sigmas,V_H = la.svd(A)
    sigmas = np.diag(sigmas)
    S_mats = [S, V_H @ S, sigmas @ V_H @ S, U @ sigmas @ V_H @ S] #lists of results
    E_mats = [E, V_H @ E, sigmas @ V_H @ E, U @ sigmas @ V_H @ E]
    titles = ["S","V_H @ S", "sigmas @ V_H @ S", "U @ sigmas @ V_H @ S"] #list of titles

    for i in range(4): #make four plots
        plt.subplot(2,2,i+1).plot(S_mats[i][0],S_mats[i][1],'b-',lw=2) #new x's and y's
        plt.subplot(2,2,i+1).plot(E_mats[i][0],E_mats[i][1],'r-',lw=2)
        plt.title(titles[i])

    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    if s > np.linalg.matrix_rank(A): #raises error if desired rank is too high
        raise ValueError("s > rank") 
    U,sigs,Vh = la.svd(A, full_matrices=False) #reduced SVD
    sigs = sigs[:s]
    U = U[:,:s]
    Vh = Vh[:s,:]
    size = sigs.size + U.size + Vh.size #get size
    A_approx = U @ np.diag(sigs) @ Vh #multiply them together to get the matrix approximation
    
    return A_approx, size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U,sigs,Vh = la.svd(A,full_matrices=False) #get svd

    if sigs[-1] > err: #check for too extreme of error
        raise ValueError("approximation cannot be done")

    r = np.min(np.where(sigs < err)) #get the rank

    return svd_approx(A,r)


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename) / 255 #normalize
    orig_size = image.size #get original size
    if image.ndim == 2: #if it's BW
        img_approx, size = svd_approx(image,s)

        plt.subplot(121).imshow(image, cmap="gray") #cmap is gray
        plt.axis("off")
        plt.subplot(122).imshow(img_approx, cmap="gray")
        plt.axis("off")
    if image.ndim == 3:
        R = image[:,:,0] #red layer
        G = image[:,:,1] #green layer
        B = image[:,:,2] #blue layer
        R, size_r = svd_approx(R,s) #get svd approx and rank
        G, size_g = svd_approx(G,s)
        B, size_b = svd_approx(B,s)

        img_approx = np.dstack((R,G,B)) #stack the approximations
        size = size_r + size_g + size_b #get size

        img_approx = np.clip(img_approx, 0, 1)

        plt.subplot(121).imshow(image)
        plt.axis("off")
        plt.subplot(122).imshow(img_approx)
        plt.axis("off")

    plt.suptitle("rank {} approximation using {} less entries".format(s,orig_size-size))
    plt.tight_layout
    plt.show()
