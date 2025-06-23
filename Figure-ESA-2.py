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

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to render text
        "font.family": "serif",  # Use serif fonts (like LaTeX default)
        "font.serif": ["Computer Modern Roman"],  # Specify LaTeX font family
    }
)
seed1 = 1
seed2 = 1
np.set_printoptions(precision=2, suppress=True, linewidth=2048)
np.random.seed(seed1)
random.seed(seed2)

n = 64
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
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
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

vals, vecs = np.linalg.eig(G.A)
val, vec = (
    vals[np.argmax(vals.real)],
    vecs[:, np.argmax(vals.real)],
)
data = {"RAND": [[], [], []], "GRDY": [[], [], []]}
while G.total_checkers() > 0:
    valpre = val
    vecpre = vec
    iterCnt += 1
    # print(iterCnt, G.total_checkers())
    swt = G.find_random_checker()
    i, j, k, l = swt
    G.switch(swt)
    vals, vecs = np.linalg.eig(G.A)
    val, vec = (
        vals[np.argmax(vals.real)],
        vecs[:, np.argmax(vals.real)],
    )
    # APost = np.linalg.eigh(G.A)[0][-1]
    x1 = 2 * (d[i] - d[j]) * (d[k] - d[l])
    x2 = 2 * (vecpre[i] - vecpre[j]) * (vecpre[k] - vecpre[l])
    y = val - valpre
    data["RAND"][0].append(x1)
    data["RAND"][1].append(x2)
    data["RAND"][2].append(y)
    mx = max(mx, x1, x2, y)
    mn = min(mn, x1, x2, y)
    print("{:8.4f} {:8.4f} {:8.4f}".format(y, x1, x2))
    label = "RAND" if iterCnt == 1 else ""
    ax[0].scatter(x2, x1, c="#1f78b4", label=label, s=8, linewidths=1, marker="x")
    ax[1].scatter(y, x1, c="#1f78b4", label=label, s=8, linewidths=1, marker="x")

iterCnt = 0
G = NetSwitch(graph)
vals, vecs = np.linalg.eig(G.A)
val, vec = (
    vals[np.argmax(vals.real)],
    vecs[:, np.argmax(vals.real)],
)
while G.total_checkers() > 0:
    valpre = val
    vecpre = vec
    iterCnt += 1
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
    # val, vec = max(zip(*np.linalg.eig(G.A)), key=lambda x: x[0].real)
    vals, vecs = np.linalg.eig(G.A)
    val, vec = (
        vals[np.argmax(vals.real)],
        vecs[:, np.argmax(vals.real)],
    )

    x1 = 2 * (d[i] - d[j]) * (d[k] - d[l])
    # print(vecpre)
    x2 = 2 * (vecpre[i] - vecpre[j]) * (vecpre[k] - vecpre[l])
    y = val - valpre
    mx = max(mx, x1, x2, y)
    mn = min(mn, x1, x2, y)
    data["GRDY"][0].append(x1)
    data["GRDY"][1].append(x2)
    data["GRDY"][2].append(y)
    print("{:8.4f} {:8.4f} {:8.4f}".format(y, x1, x2))
    label = "GRDY" if iterCnt == 1 else ""
    ax[0].scatter(
        x2,
        x1,
        edgecolor="#e31a1c",
        label=label,
        s=8,
        linewidths=1,
        marker="o",
        facecolors="none",
    )
    ax[1].scatter(
        y,
        x1,
        edgecolor="#e31a1c",
        label=label,
        s=8,
        linewidths=1,
        marker="o",
        facecolors="none",
    )


ax[0].legend(frameon=False, markerscale=2, loc="lower right")
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
pad = (mx - mn) * 0.05
ax[0].set_xlim(mn - pad, mx + pad)
ax[0].set_ylim(mn - pad, mx + pad)
ax[1].set_xlim(mn - pad, mx + pad)
ax[1].set_ylim(mn - pad, mx + pad)

ax[1].set_xlabel(
    r"$2(\mathbf{x}_{1,i}-\mathbf{x}_{1,j})(\mathbf{x}_{1,k}-\mathbf{x}_{1,l})$"
)
ax[0].set_xlabel(r"$\Delta \mu_1$")

ax[0].set_ylabel(r"$\frac{2}{\|d\|}(d_i-d_j)(d_k-d_l)$")

ax[0].set_aspect("equal")
ax[1].set_aspect("equal")

# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")

ax[0].plot([mn - pad, mx + pad], [mn - pad, mx + pad], c="k", ls="-.", lw=1, zorder=0)
ax[1].plot([mn - pad, mx + pad], [mn - pad, mx + pad], c="k", ls="-.", lw=1, zorder=0)
plt.tight_layout()

fig.savefig("ESA-Fig-2.pdf", dpi=300)

fig, ax = plt.subplots(1, 3, figsize=(8, 4))
bins = np.linspace(mn, mx, 10)

ax0_ = ax[0].twinx()
ax[0].hist(
    data["RAND"][0],
    bins=bins,
    lw=2,
    density=False,
    histtype="step",
    label="RAND",
    color="#1f78b4",
)
ax[0].bar(0, 0, edgecolor="#e31a1c", label="GRDY", facecolor="none", lw=2)
ax0_.hist(
    data["GRDY"][0],
    bins=bins,
    lw=2,
    density=False,
    histtype="step",
    label="GRDY",
    color="#e31a1c",
)
ax[0].set_xlim(mn, mx)
ax[0].set_ylim(0, len(data["RAND"][0]))
ax0_.set_ylim(0, len(data["GRDY"][0]))
ax[0].set_yticks(
    np.linspace(0, len(data["RAND"][0]), 11), np.round(np.linspace(0, 1, 11), 1)
)
ax0_.set_yticks([])

ax1_ = ax[1].twinx()
ax[1].hist(
    data["RAND"][1],
    bins=bins,
    lw=2,
    density=False,
    histtype="step",
    label="",
    color="#1f78b4",
)
ax1_.hist(
    data["GRDY"][1],
    bins=bins,
    lw=2,
    density=False,
    histtype="step",
    label="",
    color="#e31a1c",
)
ax[1].set_xlim(mn, mx)
ax[1].set_ylim(0, len(data["RAND"][1]))
ax1_.set_ylim(0, len(data["GRDY"][1]))
ax[1].set_yticks(
    np.linspace(0, len(data["RAND"][1]), 11), np.round(np.linspace(0, 1, 11), 1)
)
ax1_.set_yticks([])


ax2_ = ax[2].twinx()
ax[2].hist(
    data["RAND"][2],
    bins=bins,
    lw=2,
    density=False,
    histtype="step",
    label="",
    color="#1f78b4",
)
ax2_.hist(
    data["GRDY"][2],
    bins=bins,
    lw=2,
    density=False,
    histtype="step",
    label="",
    color="#e31a1c",
)
ax[2].set_ylim(0, len(data["RAND"][2]))
ax2_.set_ylim(0, len(data["GRDY"][2]))
ax[2].set_yticks(
    np.linspace(0, len(data["RAND"][2]), 11), np.round(np.linspace(0, 1, 11), 1)
)
ax2_.set_yticks([])

ax[2].set_xlim(mn, mx)
ax[0].legend(frameon=False)
ax[0].set_ylabel("Probability")

ax[0].set_xlabel(r"$\frac{2}{\|d\|}(d_i-d_j)(d_k-d_l)$")
ax[1].set_xlabel(
    r"$2(\mathbf{x}_{1,i}-\mathbf{x}_{1,j})(\mathbf{x}_{1,k}-\mathbf{x}_{1,l})$"
)
ax[2].set_xlabel(r"$\Delta \mu_1$")

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)

ax0_.spines["top"].set_visible(False)
ax0_.spines["right"].set_visible(False)
ax1_.spines["top"].set_visible(False)
ax1_.spines["right"].set_visible(False)
ax2_.spines["top"].set_visible(False)
ax2_.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig("ESA-Fig-2-a.pdf", dpi=300)
