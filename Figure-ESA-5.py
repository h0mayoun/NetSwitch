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

algs = ["RAND", "GRDY", "SWPC"]  # , "SWPC", "ModA-G"]
reps = [10, 10, 1]
l2levels = np.linspace(0, 1, 101)
colors = ["#e31a1c", "#33a02c", "#1f78b4", "tab:purple", "tab:green"]
# colors = cmap(np.linspace(0, 1, len(algs)))
step = [50, 10, -1]
fig, ax = plt.subplots(figsize=(6, 6))
for cnt, alg in enumerate(algs):

    for rep in range(reps[cnt]):
        for l2level in l2levels:
            print(alg, rep, l2level)
            S = NetSwitch(graph)
            data = [
                # (
                #     S.swt_done,
                #     S.lev(fast=False),
                #     S.l2(normed=True, fast=False),
                # )
            ]
            if (
                sys.argv[1] == "1"
                and rep == 0
                and not os.path.exists("result/{}/{}/".format(graph_des, alg))
            ):
                os.makedirs("result/{}/{}/".format(graph_des, alg))
                mmwrite("result/{}/{}/{}.mtx".format(graph_des, alg, S.swt_done), S.A)
            while True:
                # print(S.MScore(normed=False))
                swt_num = S.switch_A(alg=alg, count=step[cnt], l2lim=l2level)
                data.append(
                    (
                        S.swt_done,
                        S.lev(fast=False),
                        S.l2(normed=True, fast=False),
                    )
                )
                if sys.argv[1] == "1" and rep == 0:
                    mmwrite(
                        "result/{}/{}/{}.mtx".format(graph_des, alg, S.swt_done), S.A
                    )
                print(S.swt_done)
                if swt_num != -1:
                    break
            lev = np.array([i[1] for i in data])
            l2 = np.array([i[2] for i in data])

            label = "" if rep != 0 or l2level != l2levels[0] else alg
            alpha = 1
            ax.scatter(
                lev,
                l2,
                color=colors[cnt],
                label=label,
                alpha=alpha,
                s=20,
            )
            if alg == "GRDY" or alg == "RAND":
                break

if not os.path.exists("result/{}".format(graph_des)):
    os.makedirs("result/{}".format(graph_des))
ax.set_xlabel(r"$\mu_1$")
ax.set_ylabel(r"$\lambda_2$")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("ESA-Fig-5.pdf", dpi=300)
