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
import matplotlib as mpl

plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)

cmap = mpl.colormaps["tab10"]

seed1 = 1
seed2 = 1
np.random.seed(seed1)
random.seed(seed2)


n = 256
p = np.log2(n) * 1.1 / n
kn = int(np.ceil(np.log2(n)))
graphtype = "BA"
if graphtype == "ER":
    # graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
elif graphtype == "BA":
    # graph = ig.Graph.Barabasi(n=n, m=kn)
    graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)

algs = ["RAND", "GRDY", "SWPC"]  # , "SWPC", "ModA-G"]
reps = [1, 1, 1, 1]
# color = ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green"]
colors = cmap(np.linspace(0, 1, len(algs)))
step = [100, 100, 500]
fig, ax = plt.subplots(1, 2)
for cnt, alg in enumerate(algs):
    if False and os.path.isdir("result/{}/{}".format(graph_des, alg)):
        files = np.sort(
            [int(i[:-4]) for i in os.listdir("result/{}/{}".format(graph_des, alg))]
        )
        last = files[-1]
        A = mmread("result/{}/{}/{}.mtx".format(graph_des, alg, last))
        S = NetSwitch(ig.Graph.Adjacency(A))
        S.swt_done = last
    elif graphtype == "ER":
        graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    elif graphtype == "BA":
        graph = ig.Graph.Barabasi(n=n, m=kn)

    for rep in range(reps[cnt]):
        S = NetSwitch(graph)
        data = [
            (
                S.swt_done,
                S.lev(fast=False),
                S.l2(normed=True, fast=False),
                S.Mlev(normed=False, fast=False),
            )
        ]
        if sys.argv[1] == "1" and not os.path.exists(
            "result/{}/{}/".format(graph_des, alg)
        ):
            os.makedirs("result/{}/{}/".format(graph_des, alg))
            mmwrite("result/{}/{}/{}.mtx".format(graph_des, alg, S.swt_done), S.A)
        while True:
            # print(S.MScore(normed=False))
            swt_num = S.switch_A(alg=alg, count=step[cnt])
            data.append(
                (
                    S.swt_done,
                    S.lev(fast=False),
                    S.l2(normed=True, fast=False),
                    S.Mlev(normed=False, fast=False),
                )
            )
            if sys.argv[1] == "1":
                mmwrite("result/{}/{}/{}.mtx".format(graph_des, alg, S.swt_done), S.A)
            print(S.swt_done)
            if swt_num != -1:
                break
        lev = np.array([i[1] for i in data])
        l2 = np.array([i[2] for i in data])
        mlev = np.array([i[3] for i in data])

        label = "" if rep != 0 else alg
        alpha = 1
        ax[0].scatter(
            (lev - lev[0]) / lev[0],
            (l2 - l2[0]) / l2[0],
            color=colors[cnt],
            label=label,
            alpha=alpha,
            s=1,
        )
        ax[1].scatter(
            (lev - lev[0]) / lev[0],
            (mlev - mlev[0]) / mlev[0],
            color=colors[cnt],
            s=1,
        )

if not os.path.exists("result/{}".format(graph_des)):
    os.makedirs("result/{}".format(graph_des))
ax[0].set_xlabel(r"$\alpha_1$")
ax[1].set_xlabel(r"$\alpha_1$")
ax[0].set_ylabel(r"$\lambda_{N-2}$")
ax[1].set_ylabel(r"$\mu_1$")
ax[0].legend(frameon=False)
plt.tight_layout()
plt.savefig("result/{}/pareto.pdf".format(graph_des), dpi=300)
