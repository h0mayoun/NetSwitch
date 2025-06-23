import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from local.NetSwitchAlgsMod import NetSwitch

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

A = G.A
D = np.diag(G.A @ G.A)
L = D - A

P = (A.T / D).T
print(P)

print(np.linalg.eig(P))
