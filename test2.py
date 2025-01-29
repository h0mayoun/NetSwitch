import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch

np.set_printoptions(precision=2, suppress=True)


def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted descendingly
    eigVal, eigVec = np.linalg.eig(M.astype(np.float32))
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]


# Initializing NetSwitch  with an ER network
random.seed(1)
np.random.seed(1)
n = 5
p = 0.6
ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
G = NetSwitch(ERgraph)

L = np.array(ERgraph.laplacian())
A = np.array(ERgraph.get_adjacency().data)
s = np.array(np.sign(np.random.rand(n) - 0.5)).reshape(-1, 1)

D = np.diag(np.diag(G.A @ G.A))
D_halfinv = np.astype(np.diag(1 / np.sqrt(np.diag(G.A @ G.A))),np.float32)
D_half = np.astype(np.diag(np.sqrt(np.diag(G.A @ G.A))),np.float32)
I = np.eye(n)

print("D")
print(D_half)

print("L")
v, u = getLk(D - G.A, np.arange(n))
print(v, "\n", u)
print( D_halfinv @ u @ np.diag(v) @ np.linalg.inv(u) @ D_halfinv)

print("Ln")
v, u = getLk(I - D_halfinv @ G.A @ D_halfinv, np.arange(n))
print(v, "\n", u)

print("An")
v, u = getLk(D_halfinv @ G.A @ D_halfinv, np.arange(n))
print(v, "\n", u)
