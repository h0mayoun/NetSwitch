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
import time

plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)

cmap = mpl.colormaps["tab10"]

seed1 = 1
seed2 = 1
np.random.seed(seed1)
random.seed(seed2)


fig, ax = plt.subplots(2, 1, figsize=(4, 4))
figb, axb = plt.subplots(figsize=(6, 3))

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

# timedata = {"GRDY": [], "RAND": [], "BEST": []}
# for i in range(5, 9 * 2 + 1):
#     n = int(np.round(2 ** (i / 2)))
#     print(n)
#     p = np.log2(n) * 1.1 / n
#     kn = int(np.ceil(np.log2(n)))
#     graphtype = "BA"
#     if graphtype == "ER":
#         graph = ig.Graph.Erdos_Renyi(n=n, p=p)
#         graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
#     elif graphtype == "BA":
#         graph = ig.Graph.Barabasi(n=n, m=kn)
#         graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)

#     print(".", end="", flush=True)
#     S = NetSwitch(graph)
#     beginTime = time.time()
#     S.switch_A(alg="GRDY", count=-1)
#     endtime = time.time() - beginTime
#     timedata["GRDY"].append((n, endtime / S.swt_done, S.swt_done))

#     print(".", end="", flush=True)
#     S = NetSwitch(graph)
#     beginTime = time.time()
#     S.switch_A(alg="BEST", count=-1)
#     endtime = time.time() - beginTime
#     timedata["BEST"].append((n, endtime / S.swt_done, S.swt_done))

# # print(".", end="", flush=True)
# # S = NetSwitch(graph)
# # beginTime = time.time()
# # S.switch_A(alg="RAND", count=-1)
# # endtime = time.time() - beginTime
# # timedata["RAND"].append((n, endtime, S.swt_done))

# print(timedata)

# axa[0].plot([d[0] for d in timedata["GRDY"]], [d[1] for d in timedata["GRDY"]])
# axa[0].plot([d[0] for d in timedata["RAND"]], [d[1] for d in timedata["RAND"]])
# axa[0].plot([d[0] for d in timedata["BEST"]], [d[1] for d in timedata["BEST"]])

# axa[1].plot([d[0] for d in timedata["GRDY"]], [d[2] for d in timedata["GRDY"]])
# axa[1].plot([d[0] for d in timedata["RAND"]], [d[2] for d in timedata["RAND"]])
# axa[1].plot([d[0] for d in timedata["BEST"]], [d[2] for d in timedata["BEST"]])

# axa[0].set_yscale("log")
# figa.savefig("test.pdf", dpi=300)
# 0 / 0
n = 64
kn = int(np.ceil(np.log2(n)))
graphtype = "BA"
graph = ig.Graph.Barabasi(n=n, m=kn)
graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)

S = NetSwitch(graph)
data = [
    (
        S.swt_done,
        S.lev(),
        S.l2(normed=True),
    )
]
while S.total_checkers() > 0:
    print(S.swt_done, S.total_checkers())
    swt_num = S.switch_A(alg="GRDY", count=1)
    data.append(
        (
            S.swt_done,
            S.lev(),
            S.l2(normed=True),
        )
    )

swt = np.array([i[0] for i in data])
lev = np.array([i[1] for i in data])
l2 = np.array([i[2] for i in data])
ax[0].plot(swt, lev, label=r"GRDY", c="#e31a1c")
ax[1].plot(swt, l2, label=r"", c="#e31a1c")


S = NetSwitch(graph)
data = [
    (
        S.swt_done,
        S.lev(),
        S.l2(normed=True),
    )
]
while S.total_checkers() > 0:
    print(S.swt_done, S.total_checkers())
    swt_num = S.switch_A(alg="SWPC", count=1)
    data.append(
        (
            S.swt_done,
            S.lev(),
            S.l2(normed=True),
        )
    )

swt = np.array([i[0] for i in data])
lev = np.array([i[1] for i in data])
l2 = np.array([i[2] for i in data])
ax[0].plot(swt, lev, label=r"ACAR", c="#33a02c")
ax[1].plot(swt, l2, label=r"", c="#33a02c")

# S = NetSwitch(graph)
# data = [
#     (
#         S.swt_done,
#         S.lev(),
#         S.assortativity_coeff(),
#         S.deg.T @ S.A @ S.deg / (S.deg.T @ S.deg),
#     )
# ]
# while S.total_checkers() > 0:
#     # print(S.swt_done, S.total_checkers())
#     swt_num = S.switch_A(alg="RAND", count=1)
#     data.append(
#         (
#             S.swt_done,
#             S.lev(),
#             S.assortativity_coeff(),
#             S.deg.T @ S.A @ S.deg / (S.deg.T @ S.deg),
#         )
#     )

# swt = np.array([i[0] for i in data])
# lev = np.array([i[1] for i in data])
# m2 = np.array([i[2] for i in data])
# levhat = np.array([i[3] for i in data])
# axa.plot(swt, m2, label=r"RAND", c="#1f78b4")

# S = NetSwitch(graph)
# data = [
#     # (
#     #     S.swt_done,
#     #     S.lev(),
#     #     S.assortativity_coeff(),
#     #     S.deg.T @ S.A @ S.deg / (S.deg.T @ S.deg),
#     # )
# ]
# while S.total_checkers() > 0:
#     # print(S.swt_done, S.total_checkers())
#     beginTime = time.time()
#     swt_num = S.switch_A(alg="BEST", count=1)
#     endtime = time.time() - beginTime
#     data.append(
#         (
#             S.swt_done,
#             S.lev(),
#             S.assortativity_coeff(),
#             S.deg.T @ S.A @ S.deg / (S.deg.T @ S.deg),
#             endtime,
#         )
#     )
# # axa[0].set_xlim([0, S.swt_done])

# swt = np.array([i[0] for i in data])
# lev = np.array([i[1] for i in data])
# m2 = np.array([i[2] for i in data])
# print(np.mean([i[4] for i in data]))
# axa[0].plot(swt, m2, label=r"BEST", c="#1f78b4")

ax[0].set_ylabel("Spectral radius")
ax[1].set_ylabel("Algebraic\nconnectivity")
ax[1].set_xlabel("Switches done")


fig.tight_layout()

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

ax[0].legend(frameon=False, loc="lower right")

fig.savefig("ESA-Fig-1-d.pdf".format(graph_des), dpi=300)
