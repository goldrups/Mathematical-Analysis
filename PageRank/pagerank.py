# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Sam Goldrup>
<Math 347>
<1 March 2022>
"""

import numpy as np
import networkx as nx
from itertools import combinations
from scipy.sparse import linalg as la

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        A[:,A.sum(axis=0) == 0] = 1 #a better version of what i thought of doing!
        A_hat = A / A.sum(axis=0) #normalize each col
        self.A_hat = A_hat
        if labels == None:
            labels = [str(i) for i in range(0,len(A))] #handle case where there are no labels
        if (len(A)) != len(labels):
            raise ValueError("inequal number of nodes to vertices")

        self.labels = labels #save it


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        page_ranks = ((1-epsilon)/len(self.A_hat)) * np.linalg.inv(np.eye(len(self.A_hat)) - epsilon*self.A_hat) @ np.ones(len(self.A_hat)) #solve a linear system
        map = {}
        for h,j in zip(self.labels,page_ranks): #build mapping dictionary
            map[h] = j
        return map

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        B = epsilon * self.A_hat + ((1-epsilon)/len(self.A_hat))*np.ones_like(self.A_hat) #build B as specified by Eigen value problem formulation
        page_ranks = la.eigs(B,1)[1].real #eigen vectors come transposed
        page_ranks /= page_ranks.sum() #normalize
        map = {}
        for h,j in zip(self.labels,page_ranks):
            map[h] = j[0]
        return map

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        n = len(self.A_hat)
        p0 = [1/n]*n #guess
        for i in range(maxiter):
            p1 = (epsilon*self.A_hat) @ p0 + ((1-epsilon)/n) * np.ones(n) #update
            if np.linalg.norm(p1-p0,ord=1) < tol: #convergence check!
                break
            p0 = p1 #update the guess
        map = dict(zip(self.labels,p1))
        return map


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    litty = [(-val,key) for key,val in zip(d.keys(),d.values())] #we're going to do it this way to avoid some roundoff error
    litty.sort() #this is so litty tbh
    return [key[1] for key in litty] #key[1] is the position of the page

# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    with open(filename) as file:
        lines = file.readlines() #list of lines of the txt
           
    line_list = [line.strip().split('/') for line in lines] #list of lists of page ids

    labels = set() #initiliaze a set (only one element per value)
    for line in line_list:
        for id in line:
            labels.add(id)
    unique_labels = sorted(labels) #sort the set, gives a list

    n = len(unique_labels) #size of matrix
    B = np.zeros((n,n))

    map_dict = {unique_labels[j]: j for j in range(n)} #mapping dictionary, gets index

    for line in line_list:
        webpage = line[0]
        for neighbor in line[1:]:
            B[map_dict[neighbor],map_dict[webpage]] += 1 #we can use mapping dict to say where to put a 1

    page_graph = DiGraph(B,unique_labels) #make a digraph object
    epic_dict = page_graph.itersolve(epsilon=epsilon) #solve it with desired epsilon

    dank_ranks = get_ranks(epic_dict) #run get_ranks function on the dict, so dank and goated...
    return dank_ranks

# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    line_list = [] #empty list of lines
    teams = np.array([])

    with open(filename) as csv_file:
        data = csv_file.readlines() #data is this beautiful list of lines

    data = data[1:] #screw the first row (it just says win/loss)


    for line in data:
        line = line.strip()
        line = line.split(',')
        line_list.append(line)
        teams = np.concatenate((teams,line))

    games = [line for line in line_list] #gamez
    teams = list(set(list(teams))) #unique number of teams

    map_dict = {}
    for h,j in zip(teams,list(range(len(teams)))):
        map_dict[h] = j

    cols = []

    for team in teams:
        col = np.zeros(len(teams))
        for game in games: #for each game
            if team == game[0]: #if you're the loser
                col[map_dict[game[1]]] += 1 #point from loser to winner
        cols.append(col)
    
    A = np.vstack((cols)) #stack the columns togetha

    page_graph = DiGraph(A,teams) #make a digraph object 
    epic_dict = page_graph.itersolve(epsilon=epsilon) #solve to get a dict

    dank_ranks = get_ranks(epic_dict) #run get_ranks on it
    return dank_ranks


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    DG = nx.DiGraph() #use networkx library, which is commonly used by noobs who don't like mathematics

    with open("top250movies.txt",encoding="utf-8") as moviefile:
        lines = moviefile.readlines() #read in the data
    
    actors = np.array([])
    for line in lines:
        line = line.strip()
        line = line.split('/')[1:] #don't want the movie title
        for a1,a2 in combinations(line,2):
            if DG.has_edge(a2,a1): #edge points in reverse order
                DG[a2][a1]['weight'] += 1 #add weight to the edge
            else:
                DG.add_edge(a2,a1,weight=1) #or start at value of 1 if it is new

    dicto = nx.pagerank(DG,alpha=epsilon) #got the dict!, now just run get_ranks on it as shown below

    return get_ranks(dicto)

    

if __name__ == "__main__":
    # A = np.array([[0,0,0,0],[1,0,1,0],[1,0,0,1],[1,0,1,0]])
    # poop = DiGraph(A,["a","b","c","d"])
    # print(poop.A_hat, poop.labels)
    # print(poop.linsolve(),poop.eigensolve(),poop.itersolve())
    # print("0.66:",rank_websites(filename="web_stanford.txt", epsilon=0.66)[:20])
    # print("0.36:",rank_websites(filename="web_stanford.txt", epsilon=0.36)[:20])
    # print("0.11:",rank_websites(filename="web_stanford.txt", epsilon=0.11)[:20])

    # print("0.85:",rank_websites(filename="web_stanford.txt", epsilon=0.85)[:20])

    # print(rank_ncaa_teams('ncaa2010.csv', epsilon=0.11)[0:20])
    # print(rank_ncaa_teams('ncaa2010.csv', epsilon=0.82)[0:20])
    # print(rank_ncaa_teams('ncaa2010.csv', epsilon=0.50)[0:20])
    # print(rank_ncaa_teams('ncaa2010.csv', epsilon=0.90)[0:20])
    # print(rank_ncaa_teams('ncaa2010.csv', epsilon=0.59)[0:20])

    # print(rank_actors(filename="top250movies.txt", epsilon=0.85)[:20])
    pass