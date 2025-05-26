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

n = 512
p = np.log2(n) * 1.1 / n
kn = int(np.ceil(np.log2(n)))
graphtype = "BA"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)
    graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)


mx = 0
mn = 1
fig, ax = plt.subplots(figsize=(5, 5))
maxiter = 2000
cnt = 0
iterCnt = 0
G = NetSwitch(graph)
d = G.deg
d = d / np.linalg.norm(d)
cc = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]
while G.total_checkers() > 0:
    iterCnt += 1
    print(iterCnt, G.total_checkers())
    Apre = np.linalg.eigh(G.A)[0][-1]
    swt = G.find_random_checker()
    i, j, k, l = swt
    # search_block = 0

    # while True:
    #     new_k, new_l = G.largest_kl(i, j)
    #     if new_k == k and new_l == l:
    #         search_block += 1
    #         if search_block == 2:
    #             break
    #     else:
    #         k, l = new_k, new_l
    #         search_block = 0
    #     new_i, new_j = G.largest_ij(k, l)
    #     if new_i == i and new_j == j:
    #         search_block += 1
    #         if search_block == 2:
    #             break
    #     else:
    #         i, j = new_i, new_j
    #         search_block = 0
    # swt = (i, j, k, l)
    G.switch(swt)
    APost = np.linalg.eigh(G.A)[0][-1]
    x = 2 * (d[i] - d[j]) * (d[k] - d[l])
    y = APost - Apre
    mx = max(mx, x, y)
    mn = min(mn, x, y)
    print(x, y, APost)
    label = "RAND" if iterCnt == 1 else ""
    ax.scatter(x, y, c="#33a02c", label=label, s=4, linewidths=1, marker="x")

iterCnt = 0
G = NetSwitch(graph)
while G.total_checkers() > 0:
    iterCnt += 1
    print(iterCnt, G.total_checkers())
    Apre = np.linalg.eigh(G.A)[0][-1]
    swt = G.find_random_checker()
    i, j, k, l = swt
    search_block = 0

    while True:
        new_k, new_l = G.largest_kl(i, j)
        if new_k == k and new_l == l:
            search_block += 1
            if search_block == 2:
                break
        else:
            k, l = new_k, new_l
            search_block = 0
        new_i, new_j = G.largest_ij(k, l)
        if new_i == i and new_j == j:
            search_block += 1
            if search_block == 2:
                break
        else:
            i, j = new_i, new_j
            search_block = 0
    swt = (i, j, k, l)
    G.switch(swt)
    APost = np.linalg.eigh(G.A)[0][-1]
    x = 2 * (d[i] - d[j]) * (d[k] - d[l])
    y = APost - Apre
    mx = max(mx, x, y)
    mn = min(mn, x, y)
    print(x, y, APost)
    label = "GRDY" if iterCnt == 1 else ""
    ax.scatter(
        x,
        y,
        edgecolor="#e31a1c",
        label=label,
        s=4,
        linewidths=1,
        marker="o",
        facecolors="none",
    )

ax.legend(frameon=False, markerscale=2, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
pad = (mx - mn) * 0.05
ax.set_xlim(mn - pad, mx + pad)
ax.set_ylim(mn - pad, mx + pad)
ax.set_xlabel(r"$2(d_i-d_j)(d_k-d_l)$")
ax.set_ylabel(r"$\Delta \mu_1$")
ax.set_aspect("equal")
ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], c="k", ls="-.", lw=1, zorder=0)
plt.tight_layout()
fig.savefig("ESA-Fig-2.pdf", dpi=300)
