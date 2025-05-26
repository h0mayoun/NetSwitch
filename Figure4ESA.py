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


# plt.rcParams.update(
#     {
#         "text.usetex": True,  # Use LaTeX to render text
#         "font.family": "serif",  # Use serif fonts (like LaTeX default)
#         "font.serif": ["Computer Modern Roman"],  # Specify LaTeX font family
#     }
# )

seed1 = 1
seed2 = 1
np.set_printoptions(precision=2, suppress=True, linewidth=2048)
np.random.seed(seed1)
random.seed(seed2)

n = 32
p = np.log2(n) * 1.1 / n
kn = int(np.ceil(np.log2(n)))
graphtype = "BA"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)
    graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)
S = NetSwitch(graph)
maxiter = 10000
if not os.path.exists("result/{}".format(graph_des)):
    os.makedirs("result/{}".format(graph_des))

# layout = graph.layout("grid")
visual_style = {
    "vertex_size": 10,
    "vertex_color": "skyblue",
    "vertex_label": None,  # Hide labels
    "edge_color": "black",
    "vertex_frame_width": 0,
    "edge_width": 0.5,
    "bbox": (400, 400),  # Size of the plot canvas
    "margin": 20,
    "layout": "auto",
}

fig, ax = plt.subplots(3, 6, figsize=(20, 10))
ax[0][1].axis("off")
ax[2][1].axis("off")
ax[0][4].axis("off")
ax[2][4].axis("off")

ig.plot(graph, target=ax[1][1], **visual_style)
ax[1][1].set_title("O")
G = NetSwitch(graph)
G.plotAdjacencyImage(ax[1][4], -1 * np.ones((n, 1)))
ax[1][4].set_title("O")
mmwrite("result/{}/O.mtx".format(graph_des), G.A)
cnt = 0
iterCnt = 0
G = NetSwitch(graph)
G.sort_adj(ordr=getStat(G.A)[1][0])
print(getStat(G.A)[0][0], end=" ")
while iterCnt < maxiter:
    a1Pre = getStat(G.A)[0][0]

    swt = G.find_random_checker_mod()
    if swt == (-1, -1, -1, -1):
        iterCnt += 1
        continue
    G.switch(swt, update_N=False)

    a1Post = getStat(G.A)[0][0]

    if a1Post <= a1Pre:
        cnt += 1
        G.update_N(swt)
        print(".", end="", flush=True)
        iterCnt = 0
    else:
        iterCnt += 1
        G.switch(swt, update_N=False)

ig.plot(
    ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED), target=ax[0][0], **visual_style
)
print(getStat(G.A)[0][0], end=" ")
print()
ax[0][0].set_title("A-")
G.sort_adj(ordr=getStat(G.A)[1][0])
G.plotAdjacencyImage(ax[0][3], -1 * np.ones((n, 1)))
ax[0][3].set_title("A-")
mmwrite("result/{}/A-.mtx".format(graph_des), G.A)

cnt = 0
iterCnt = 0
G = NetSwitch(graph)
G.sort_adj(ordr=getStat(G.A)[1][0])
print(getStat(G.A)[0][0], end=" ")
while iterCnt < maxiter:
    a1Pre = getStat(G.A)[0][0]

    swt = G.find_random_checker_mod()
    if swt == (-1, -1, -1, -1):
        iterCnt += 1
        continue
    G.switch(swt, update_N=False)

    a1Post = getStat(G.A)[0][0]

    if a1Post >= a1Pre:
        cnt += 1
        G.update_N(swt)
        print(".", end="", flush=True)
        iterCnt = 0
    else:
        iterCnt += 1
        G.switch(swt, update_N=False)

ig.plot(
    ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED), target=ax[0][2], **visual_style
)
print(getStat(G.A)[0][0], end=" ")
print()
ax[0][2].set_title("A+")
G.sort_adj(ordr=getStat(G.A)[1][0])
G.plotAdjacencyImage(ax[0][5], -1 * np.ones((n, 1)))
ax[0][5].set_title("A+")
mmwrite("result/{}/A+.mtx".format(graph_des), G.A)

cnt = 0
iterCnt = 0
G = NetSwitch(graph)
G.sort_adj(ordr=getStat(G.A)[1][1])
print(getStat(G.A)[0][1], end=" ")
while iterCnt < maxiter * 10:
    l2Pre = getStat(G.A)[0][1]

    swt = G.find_random_checker_mod()
    if swt == (-1, -1, -1, -1):
        iterCnt += 1
        continue
    G.switch(swt, update_N=False)

    l2Post = getStat(G.A)[0][1]

    if l2Pre >= l2Post:
        cnt += 1
        G.update_N(swt)
        print(".", end="", flush=True)
        iterCnt = 0
    else:
        iterCnt += 1
        G.switch(swt, update_N=False)

ig.plot(
    ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED), target=ax[1][2], **visual_style
)
print(getStat(G.A)[0][1], end=" ")
print()
ax[1][2].set_title("L-")
G.sort_adj(ordr=getStat(G.A)[1][1])
G.plotAdjacencyImage(ax[1][5], -1 * np.ones((n, 1)))
ax[1][5].set_title("L-")
mmwrite("result/{}/L-.mtx".format(graph_des), G.A)

cnt = 0
iterCnt = 0
G = NetSwitch(graph)
G.sort_adj(ordr=getStat(G.A)[1][1])
print(getStat(G.A)[0][1], end=" ")
while iterCnt < maxiter * 10:
    l2Pre = getStat(G.A)[0][1]

    swt = G.find_random_checker_mod()
    if swt == (-1, -1, -1, -1):
        iterCnt += 1
        continue
    G.switch(swt, update_N=False)

    l2Post = getStat(G.A)[0][1]

    if l2Pre <= l2Post:
        cnt += 1
        G.update_N(swt)
        print(".", end="", flush=True)
        iterCnt = 0
    else:
        iterCnt += 1
        G.switch(swt, update_N=False)

ig.plot(
    ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED), target=ax[1][0], **visual_style
)
print(getStat(G.A)[0][1], end=" ")
print()
G.sort_adj(ordr=getStat(G.A)[1][1])
ax[1][0].set_title("L+")
G.plotAdjacencyImage(ax[1][3], -1 * np.ones((n, 1)))
ax[1][3].set_title("L+")
mmwrite("result/{}/L+.mtx".format(graph_des), G.A)

cnt = 0
iterCnt = 0
G = NetSwitch(graph)
G.sort_adj(ordr=getStat(G.A)[1][2])
print(getStat(G.A)[0][2], end=" ")
while iterCnt < maxiter:
    m1Pre = getStat(G.A)[0][2]

    swt = G.find_random_checker_mod()
    if swt == (-1, -1, -1, -1):
        iterCnt += 1
        continue
    G.switch(swt, update_N=False)

    m1Post = getStat(G.A)[0][2]

    if m1Pre >= m1Post:
        cnt += 1
        G.update_N(swt)
        print(".", end="", flush=True)
        iterCnt = 0
    else:
        iterCnt += 1
        G.switch(swt, update_N=False)

ig.plot(
    ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED), target=ax[2][0], **visual_style
)
print(getStat(G.A)[0][2], end=" ")
print()
ax[2][0].set_title("M-")
G.sort_adj(ordr=getStat(G.A)[1][2])
G.plotAdjacencyImage(ax[2][3], -1 * np.ones((n, 1)))
ax[2][3].set_title("M-")
mmwrite("result/{}/M-.mtx".format(graph_des), G.A)

cnt = 0
iterCnt = 0
G = NetSwitch(graph)
G.sort_adj(ordr=getStat(G.A)[1][2])
print(getStat(G.A)[0][2], end=" ")
while iterCnt < maxiter:
    m1Pre = getStat(G.A)[0][2]

    swt = G.find_random_checker_mod()
    if swt == (-1, -1, -1, -1):
        iterCnt += 1
        continue
    G.switch(swt, update_N=False)

    m1Post = getStat(G.A)[0][2]

    if m1Pre <= m1Post:
        cnt += 1
        G.update_N(swt)
        print(".", end="", flush=True)
        iterCnt = 0
    else:
        iterCnt += 1
        G.switch(swt, update_N=False)

ig.plot(
    ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED), target=ax[2][2], **visual_style
)
print(getStat(G.A)[0][2], end=" ")
print()
ax[2][2].set_title("M+")
G.sort_adj(ordr=getStat(G.A)[1][2])
G.plotAdjacencyImage(ax[2][5], -1 * np.ones((n, 1)))
ax[2][5].set_title("M+")
mmwrite("result/{}/M+.mtx".format(graph_des), G.A)

# A+
plt.tight_layout()
plt.show()
