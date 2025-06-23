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




plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)


seed1 = 1
seed2 = 1
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

# colors = cmap(np.linspace(0, 1, len(algs)))
step = [1, 1, 1]
fig, ax = plt.subplots(figsize=(4, 4))
for cnt, (alg, lbl) in enumerate(zip(algs, lbls)):

    for rep in range(reps[cnt]):
        for l2level in l2levels:
            print(alg, rep, l2level)
            S = NetSwitch(graph)
            data = [
                (
                    S.swt_done,
                    S.lev(fast=False),
                    S.l2(normed=True, fast=False),
                )
            ]
            x = np.real(data[-1][1])
            y = np.real(data[-1][2])
            paretoFronts[alg] = add_point_to_pareto_front(paretoFronts[alg], (x, y))
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
                x = np.real(data[-1][1])
                y = np.real(data[-1][2])
                paretoFronts[alg] = add_point_to_pareto_front(paretoFronts[alg], (x, y))
                print(S.swt_done, x, y)
                if swt_num != -1:
                    break
            lev = np.array([i[1] for i in data])
            l2 = np.array([i[2] for i in data])
            # label = "" if rep != 0 or l2level != l2levels[0] else lbl
            # alpha = 0.01
            # fc = "none" if alg != "RAND" else colors[cnt]
            # ax.scatter(
            #     lev,
            #     l2,
            #     edgecolor=colors[cnt],
            #     label="",
            #     alpha=alpha,
            #     s=10,
            #     linewidths=1,
            #     marker=marker[cnt],
            #     facecolors=fc,
            # )
            if alg == "GRDY" or alg == "RAND":
                break

S = NetSwitch(graph)
mu_1 = S.lev()
la_2 = S.l2(normed=True)

with open("result/{}/pareto.pkl".format(graph_des), "wb") as f:
    pickle.dump(paretoFronts, f)

for cnt, alg in enumerate(algs):
    paretoFronts[alg].sort(key=lambda p: p[0])
    x = [i[0] / mu_1 for i in paretoFronts[alg]]
    y = [i[1] / la_2 for i in paretoFronts[alg]]
    ax.step(x, y, c=colors[cnt], where="post", label=lbls[cnt])
    # ax.scatter(
    #     [],
    #     [],
    #     edgecolor=colors[cnt],
    #     marker=marker[cnt],
    #     s=5,
    #     label=lbls[cnt],
    #     linewidths=1,
    #     facecolors=fc,
    # )
ax.set_xlim(1, 1.25)
ax.set_ylim(0, 1.25)
ax.set_yticks(np.linspace(0, 1.25, 6))
# ax.plot([], [], c="k", label="Pareto\nFrontier")
if not os.path.exists("result/{}".format(graph_des)):
    os.makedirs("result/{}".format(graph_des))
ax.set_xlabel(r"$\frac{\mu_1}{\mu_1^{(0)}}$")
ax.set_ylabel(r"$\frac{\lambda_2}{\lambda_2^{(0)}}$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, markerscale=2)
plt.tight_layout()
plt.savefig("ESA-Fig-5.pdf", dpi=300)
