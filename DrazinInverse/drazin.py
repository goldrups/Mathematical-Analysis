# drazin.py
"""Volume 1: The Drazin Inverse.
<Sam Goldrup>
<MATH 347>
<8 April 2022>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse.csgraph import laplacian


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    Ak = np.eye(len(A),dtype=np.float) 
    for i in range(k): #A raised to the kth power
        Ak = Ak @ A
    if np.allclose(A @ Ad, Ad @ A) and np.allclose(Ad @ A @ Ad,Ad) and np.allclose(Ak @ A @ Ad, Ak): #check all 3 conditions
        return True
    else:
        return False


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    n = len(A)
    f = lambda x: abs(x) > tol #two ways to sort
    g = lambda x: abs(x) <= tol
    T1,Q1,k1 = la.schur(A, sort=f) #two schur decompositions
    T2,Q2,k2 = la.schur(A, sort=g)
    U = np.hstack((Q1[:,:k1],Q2[:,:n-k1]))
    U_inv = np.linalg.inv(U) #invert!
    V = U_inv @ A @ U
    Z = np.zeros((n,n),dtype=np.float) #float!
    if k1 != 0: #if clause!
        M_inv = np.linalg.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inv
    return U@Z@U_inv #matrix multiply them together!


# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n = len(A)
    I = np.eye(n,dtype=np.float) #identity matrix
    L = laplacian(A) #laplacian
    rows = []
    for j in range(n):
        L_j = L.copy() #copy the L
        L_j[j] = I[j] #jth row to swaparoo
        L_j_draz = drazin_inverse(L_j)
        row = np.array([L_j_draz[i,i] if i != j else 0 for i in range(n)]) #0's on diagonal baby
        rows.append(row)
    R = np.vstack((rows)) #build matrix
    return R


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        with open(filename, 'r') as social:
            data = social.readlines()
        lines = [line.strip().split() for line in data]
        lines = np.array([line[0].split(',') for line in lines])
        names = {name for line in lines for name in line}

        #nodes.sort() #sort the list of network members
        n = len(names) #size of network
        self.name_to_idx = {name:i for i,name in enumerate(list(names))} #dictionary mapping names to indices in the list
        A = np.zeros((n,n),dtype=np.float) #Adjacency matrix
        for line in lines:
            i = self.name_to_idx[line[0]] #get i,j
            j = self.name_to_idx[line[1]]
            A[i,j] = 1 #increment A[i,j]
            A[j,i] = 1
        
        res_mat = effective_resistance(A) #get effective resistence
        self.names, self.A, self.R = list(names), A, res_mat
        


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        R = self.R
        R = R * (np.ones_like(self.A,dtype=np.float) - self.A) #cleaning!
        if node == None:
            minval = np.min(R[np.nonzero(R)]) #get minimal value
            loc = np.where(R==minval) #get indices of all the min values
            return self.names[loc[0][0]],self.names[loc[1][0]]
        else:
            if node in self.names:
                i = self.name_to_idx[node] #get index
                col = R[:,i] #get that column
                minval = np.min(col[np.nonzero(col)]) #find minny
                loc = np.where(R==minval)
                return self.names[loc[0][0]]
            else:
                raise ValueError("Node not in network")


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        if node1 in self.names:
            if node2 in self.names:
                i,j = self.name_to_idx[node1],self.name_to_idx[node2]
                self.A[i,j] = 1 #update the adjacency matrices
                self.A[j,i] = 1
                self.R = effective_resistance(self.A) #get the resistances
            else:
                raise ValueError("second node not in names")
        else:
            raise ValueError("first node not in names")

if __name__ == "__main__":
    # A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
    # Ad = np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
    # print(is_drazin(A,Ad,1))
    # B = np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
    # Bd = np.array([[0,0,0],[0,0,0],[0,0,0]])
    # print(is_drazin(B,Bd,3))
    # print(is_drazin(A,drazin_inverse(A),1))
    # print(is_drazin(B,drazin_inverse(B),3))
    # C = np.random.random((5,5))
    # k = index(C)
    # Cd = drazin_inverse(C)
    # I = np.eye(len(Cd))
    # print(is_drazin(C,Cd+I,k))
    # A = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
    # print(effective_resistance(A))
    # A = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
    # print(effective_resistance(A))
    # B = np.array([[0,1],[1,0]])
    # print(effective_resistance(B))
    # C = np.array([[0,1,1],[1,0,1],[1,1,0]])
    # print(effective_resistance(C))
    # D = np.array([[0,3],[3,0]])
    # print(effective_resistance(D))
    # E = np.array([[0,2],[2,0]])
    # print(effective_resistance(E))
    # F = np.array([[0,4],[4,0]])
    # print(effective_resistance(F))
    pass
