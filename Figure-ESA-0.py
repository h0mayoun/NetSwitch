from NetSwitchAlgsMod import *
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

np.set_printoptions(precision=2, suppress=True, linewidth=2048)
plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to render text
        "font.family": "serif",  # Use serif fonts (like LaTeX default)
        "font.serif": ["Computer Modern Roman"],  # Specify LaTeX font family
    }
)
visual_style = {
    "vertex_size": 5,
    "vertex_color": "black",
    "vertex_label": None,  # Hide labels
    "edge_color": "grey",
    "vertex_frame_width": 0,
    "edge_width": 0.1,
    "bbox": (400, 400),  # Size of the plot canvas
    "margin": 20,
    "layout": "auto",
}


def getStat(A):
    d = np.sum(A, axis=0)
    D = np.diag(d)
    B = d @ d.T / sum(d)
    val, vec = np.linalg.eigh(A)
    aval, avec = val[-1], vec[:, -1]

    val, vec = np.linalg.eigh(D - A)
    lval, lvec = val[1], vec[:, 1]

    val, vec = np.linalg.eigh(A - B)
    mval, mvec = val[-1], vec[:, -1]

    return np.array(
        [
            aval,
            lval,
            mval,
        ]
    ), (avec, lvec, mvec)


directory = "result/{}".format(sys.argv[1])
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
fig2, ax2 = plt.subplots(3, 3, figsize=(10, 10))
ax[0][1].axis("off")
ax[2][1].axis("off")

A = mmread("{}/O.mtx".format(directory))
n = A.shape[0]
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[1][1], **visual_style)
print("O", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))

G.sort_adj(ordr=getStat(G.A)[1][0])
G.plotAdjacencyImage(ax2[0][1], -1 * np.ones((n, 1)))
G.sort_adj(ordr=getStat(G.A)[1][1])
G.plotAdjacencyImage(ax2[1][1], -1 * np.ones((n, 1)))
G.sort_adj(ordr=getStat(G.A)[1][2])
G.plotAdjacencyImage(ax2[2][1], -1 * np.ones((n, 1)))

A = mmread("{}/A-.mtx".format(directory))
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[0][0], **visual_style)
print("A-", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))
G.sort_adj(ordr=getStat(G.A)[1][0])
G.plotAdjacencyImage(ax2[0][0], -1 * np.ones((n, 1)))

A = mmread("{}/A+.mtx".format(directory))
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[0][2], **visual_style)
print("A+", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))
G.sort_adj(ordr=getStat(G.A)[1][0])
G.plotAdjacencyImage(ax2[0][2], -1 * np.ones((n, 1)))

A = mmread("{}/L+.mtx".format(directory))
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[1][0], **visual_style)
print("L+", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))
G.sort_adj(ordr=getStat(G.A)[1][1])
G.plotAdjacencyImage(ax2[1][0], -1 * np.ones((n, 1)))

A = mmread("{}/L-.mtx".format(directory))
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[1][2], **visual_style)
print("L-", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))
G.sort_adj(ordr=getStat(G.A)[1][1])
G.plotAdjacencyImage(ax2[1][2], -1 * np.ones((n, 1)))

A = mmread("{}/M-.mtx".format(directory))
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[2][0], **visual_style)
print("M-", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))
G.sort_adj(ordr=getStat(G.A)[1][2])
G.plotAdjacencyImage(ax2[2][0], -1 * np.ones((n, 1)))

A = mmread("{}/M+.mtx".format(directory))
ig.plot(ig.Graph.Adjacency(A, mode=ig.ADJ_UNDIRECTED), target=ax[2][2], **visual_style)
print("M+", getStat(A)[0])
G = NetSwitch(ig.Graph.Adjacency(A))
G.sort_adj(ordr=getStat(G.A)[1][2])
G.plotAdjacencyImage(ax2[2][2], -1 * np.ones((n, 1)))

plt.tight_layout()
fig.savefig("ESA-Fig-1.pdf", dpi=1000)
fig2.savefig("ESA-Fig-2.pdf", dpi=1000)
