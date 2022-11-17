# image_segmentation.py
"""Volume 1: Image Segmentation.
Samuel Goldrup
Math 345
9 November 2021
"""

import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import scipy

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.diag(A.sum(axis=1)) #make row sums of A the diagonals of D
    return D-A #L=D-A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    eigvals = sorted(np.real(scipy.linalg.eigvals(laplacian(A)))) #get second smallest eigen value
    zeros = [k for k in eigvals if abs(k) < tol] #get the number of connected components
    
    return len(zeros), eigvals[1]



# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename) / 255
        if self.image.ndim == 3: #if its a colored
            self.brightness = self.image.mean(axis=2)
            self.flat = np.ravel(self.brightness) #flattened
            self.m, self.n, _ = self.image.shape[0], self.image.shape[1], self.image.shape[2] #m and n, don't care about third thing
        elif self.image.ndim == 2: #if it is BW
            self.brightness = self.image
            self.m, self.n = self.image.shape[0], self.image.shape[1] #get m and n
            self.flat = np.ravel(self.brightness) #get flattened
    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.image.ndim == 3:
            plt.imshow(self.image) #color plot
            plt.axis("off")
        elif self.image.ndim == 2:
            plt.imshow(self.image, cmap == "gray") #bw plot
            plt.adxis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        m, n = self.m, self.n
        A = scipy.sparse.lil_matrix((m*n, m*n))
        D = np.zeros(m*n)
        for i in range(m*n):
            neighbors, distances = get_neighbors(i,r,self.m,self.n) #get the neighbors within the radius r
            weights = np.zeros(len(neighbors))
            for j in range(len(distances)):
                weights[j] = np.exp(-(abs(self.flat[neighbors[j]]-self.flat[i])/sigma_B2) - (distances[j]/sigma_X2)) #use the given math eqn
            A[i,neighbors] = weights
            D[i] = np.sum(weights) #sum of weights is diagonal entry
        A = A.tocsc() #makes the computation easier
        return A,D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = scipy.sparse.csgraph.laplacian(A) #element wise exponentiation
        D = scipy.sparse.diags(1/np.sqrt(D))
        M = D @ L @ D #D^-1/2 L D^-1/2
        _, e_vecs = scipy.sparse.linalg.eigsh(M,k=2,which="SM")
        e_vec = e_vecs[:,1].reshape((self.m,self.n)) #get the eigen vectors
        mask = e_vec > 0 #make mask that evaluates to true if value in vector is greater than 0
        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A,D = self.adjacency(r,sigma_B,sigma_X)
        mask = self.cut(A,D)
        neg_mask = ~mask
        if self.image.ndim == 2: #if its BW
            plt.subplot(131).imshow(self.image, cmap = "gray")
            plt.axis("off")
            plt.subplot(132).imshow(self.image * mask, cmap = "gray")
            plt.axis("off")
            plt.subplot(133).imshow(self.image * neg_mask, cmap = "gray")
            plt.axis("off")
        if self.image.ndim == 3: #if its colored
            mask = np.dstack((mask,mask,mask)) #stack
            neg_mask = np.dstack((neg_mask,neg_mask,neg_mask)) #stack the masks
            plt.subplot(131).imshow(self.image, cmap = "viridis")
            plt.axis("off")
            plt.subplot(132).imshow(self.image * mask, cmap = "viridis")
            plt.axis("off")
            plt.subplot(133).imshow(self.image * neg_mask, cmap = "viridis")
            plt.axis("off")
        

        plt.tight_layout()
        plt.show()