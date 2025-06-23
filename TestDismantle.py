from SIS import *
from OptiMantle import *
from local.NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle
import sys

np.set_printoptions(precision=1, suppress=True)

graphDef = sys.argv[1]
id1, id2 = sys.argv[2], sys.argv[3]
labels = ["ORG", "ACAS", "GRDY"]
colors = ["tab:blue", "tab:orange", "tab:green"]

Gs = [
    read_Graph("result/{}/SWPC/0.mtx".format(graphDef)),
    read_Graph("result/{}/SWPC/{}.mtx".format(graphDef, id1)),
    read_Graph("result/{}/GRDY/{}.mtx".format(graphDef, id2)),
]
Gs = [ig.Graph.Adjacency(g) for g in Gs]

fig, ax = plt.subplots(2, 1, figsize=(6, 6))

color = mpl.colormaps.get_cmap("tab20")
lss = ["-", "--", ":"]
for graph, glabel, ls in zip(Gs, labels, lss):
    # print(graph, glabel)
    print(glabel)
    G = OptiMantle(graph)
    lcc, con, timelog = G.dismantle(alg="deg-a")
    ax[0].plot(lcc, color=color(0), label="", ls=ls)
    ax[1].plot(con, color=color(0), label="", ls=ls)

    G = OptiMantle(graph)
    lcc, con, timelog = G.dismantle(alg="deg")
    ax[0].plot(lcc, color=color(1 / 20), label="", ls=ls)
    ax[1].plot(con, color=color(1 / 20), label="", ls=ls)

    G = OptiMantle(graph)
    lcc, con, timelog = G.dismantle(alg="betw")
    ax[0].plot(lcc, color=color(2 / 20), label="", ls=ls)
    ax[1].plot(con, color=color(2 / 20), label="", ls=ls)

    G = OptiMantle(graph)
    lcc, con, timelog = G.dismantle(alg="betw-a")
    ax[0].plot(lcc, color=color(3 / 20), label="", ls=ls)
    ax[1].plot(con, color=color(3 / 20), label="", ls=ls)

    # avg = np.zeros((graph.vcount(), 2))
    # maxIter = 100
    # for i in range(maxIter):
    #     G = OptiMantle(graph)
    #     lcc, con, timelog = G.dismantle(alg="rand")
    #     ax[0].plot(lcc, color=color, label="", ls="-.", alpha=1 / maxIter)
    #     ax[1].plot(con, color=color, label="", ls="-.", alpha=1 / maxIter)
    #     avg[:, 0] = avg[:, 0] + 1 / maxIter * lcc
    #     avg[:, 1] = avg[:, 0] + 1 / maxIter * con

    # ax[0].plot(avg[:, 0], color=color, label="", ls="-.")
    # ax[1].plot(avg[:, 1], color=color, label="", ls="-.")

plt.savefig("TestDismantle.pdf", dpi=300)
