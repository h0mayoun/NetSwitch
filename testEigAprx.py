from local.NetSwitchAlgsMod import *
import pickle
import random
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import copy
import os
from scipy.io import mmread, mmwrite
from readGraph import read_Graph
import sys
import matplotlib as mpl

np.set_printoptions(precision = 1,suppress=True)
def k_smallest_eigenpairs(A, k):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort indices of eigenvalues by magnitude
    sorted_indices = np.argsort(np.abs(eigenvalues))
    
    # Get the k smallest eigenvalues and corresponding eigenvectors
    smallest_values = eigenvalues[sorted_indices[:k]]
    smallest_vectors = eigenvectors[:, sorted_indices[:k]]
    
    # Return as list of (eigenvalue, eigenvector) pairs
    return smallest_values, smallest_vectors

# plt.rcParams.update(
#     {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
# )
cmap = mpl.colormaps["viridis"]
color = cmap(np.linspace(0,1,100))
seed1 = 1
seed2 = 1
np.random.seed(seed1)
random.seed(seed2)

n = 256
p = np.log2(n) * 1.1 / n
kn = int(np.ceil(np.log2(n)))
graphtype = "BA"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)
    graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)
    

fix,ax = plt.subplots()
S = NetSwitch(graph)
k=16
eigval, eigvec = k_smallest_eigenpairs(S.laplacian(),k)
cnt = 1
while True:
    A_pre = copy.copy(S.A)
    swt_num = S.switch_A(alg="GRDY", count=1)
    deltaA = S.A-A_pre
    eigval_,eigvec_ = np.linalg.eig(np.diag(eigval)+eigvec.T@deltaA@eigvec)
    eigvec_ = eigvec@eigvec_
    
    idx = np.argsort(np.abs(eigval_))
    eigval_ = eigval_[idx]
    eigvec_ = eigvec_[:,idx]
    
    eigval, eigvec = eigval_, eigvec_
    if cnt%100==0:
        print(eigval_)
        print(k_smallest_eigenpairs(S.laplacian(),k)[0],"\n")
    
    #eigval = np.sort(np.linalg.eigvals(S.laplacian()))[:10]
    #ax.plot(eigval,color = color[cnt])
    cnt += 1
    print(S.swt_done)
    if swt_num != -1:
        break
plt.title(r"Laplacian's eigenvalue")
plt.savefig("eigTest.pdf",dpi = 300)