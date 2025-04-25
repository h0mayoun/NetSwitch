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

seed1 = 1
seed2 = 1
np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)

np.random.seed(seed1)
random.seed(seed2)

file = [
    "email-enron-only.mtx",
    "reptilia-tortoise-network-bsv.edges",
    "inf-USAir97.mtx",
    "aves-wildbird-network.edges",
    "ca-netscience.mtx",
    "ia-radoslaw-email.edges",
    "ca-CSphd.mtx",
    "ca-GrQc.mtx",
]

# n = 256
# p = np.log2(n) * 1.1 / n
# kn = 4
# graphtype = "BA"
# if graphtype == "ER":
#     graph = ig.Graph.Erdos_Renyi(n=n, p=p)
#     graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
# elif graphtype == "BA":
#     graph = ig.Graph.Barabasi(n=n, m=kn)
#     graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)
# S = NetSwitch(graph)

filenum = 4
A = read_Graph("graphs/" + file[filenum])
S = NetSwitch(ig.Graph.Adjacency(A))
graph_des = file[filenum]
# A = read_Graph("graphs/reptilia-tortoise-network-bsv.edges")
# S = NetSwitch(ig.Graph.Adjacency(A))

# A = read_Graph("graphs/inf-USAir97.mtx")
# S = NetSwitch(ig.Graph.Adjacency(A))
# A = read_Graph("graphs/ia-radoslaw-email.edges",meanDeg=20)
# S = NetSwitch(ig.Graph.Adjacency(A))
# print(np.sum(A)/S.n)
# fig, ax = plt.subplots(2, 2, figsize=(3, 3))
# S.plotAdjacencyImage(ax[0,0])
# plt.savefig("img.png",dpi=1000)
# #print(A)
# 0/0
# fig, ax = plt.subplots(2, 3, figsize=(9, 3))
# S.plotAdjacencyImage(ax[0,0])
# S.plotNetSwitchGraph(ax[0,1])
# ax[0,1].axis('equal')
# ax[0,2].plot(S.base_mod)
# modAprx = np.zeros(S.n)
# degVec = S.deg.reshape(1, -1)
# for u in range(1, S.n):
#     s = np.array(
#         [
#             (
#                 -S.deg[i] / np.sqrt(2 * S.m * S.n)
#                 if i < u
#                 else S.deg[i] / np.sqrt(2 * S.m * S.n)
#             )
#             for i in range(S.n)
#         ]
#     ).reshape(1, -1)
#     modAprx[u] = np.mean(degVec) - np.sum(s.T @ s)
# fig, ax = plt.subplots(2, 3, figsize=(9, 9))
# S.switch_A(alg="GRDY")
# S.plotAdjacencyImage(ax[1, 0])
# S.plotNetSwitchGraph(ax[1, 1])
# ax[1, 1].axis("equal")
# ax[1, 2].plot(S.base_mod)
# ax[1, 2].plot(modAprx)
# ax[1, 2].plot(S.M_limit)

# plt.savefig("test.png", dpi=1000)
# 0 / 0
fig = plt.figure(figsize=(9, 9))
plt.suptitle(graph_des)
ax1, ax2, ax3 = (
    fig.add_subplot(3, 3, 1),
    fig.add_subplot(3, 3, 3),
    fig.add_subplot(3, 3, 4),
)
S.plotAdjacencyImage(ax1)
S.plotNetSwitchGraph(ax2)
ax3.plot(S.base_mod)
ax2.axis("equal")


data = [
    (
        S.swt_done,
        S.lev(fast=False),
        S.l2(normed=True, fast=False),
        S.Mlev(normed=False, fast=False),
        S.MScore(normed=False),
        S.L2Score(normed=True),
    )
]
alg = "GRDY"
while True:
    swt_num = S.switch_A(alg=alg, count=1000)
    data.append(
        (
            S.swt_done,
            S.lev(fast=False),
            S.l2(normed=True, fast=False),
            S.Mlev(normed=False, fast=False),
            S.MScore(normed=False),
            S.L2Score(normed=True),
        )
    )

    if not os.path.exists("result/{}/{}/".format(graph_des, alg)):
        os.makedirs("result/{}/{}/".format(graph_des, alg))

    mmwrite("result/{}/{}/{}.mtx".format(graph_des, alg, S.swt_done), S.A)
    print(S.swt_done)
    if swt_num != -1:
        break

print("did ", (S.swt_done), " switches")
print("rejected ", (S.swt_rejected), " switches")

ax4, ax5, ax6 = (
    fig.add_subplot(3, 3, 2),
    fig.add_subplot(3, 3, 6),
    fig.add_subplot(3, 3, 5),
)
S.plotAdjacencyImage(ax4)
S.plotNetSwitchGraph(ax5)
ax6.plot(S.base_mod)
ax6.plot(S.M_ub)
ax6.plot(S.M_lb)
ax5.axis("equal")


ax7 = fig.add_subplot(3, 1, 3)
ax7.plot(
    [i[0] for i in data],
    [100 * (i[1] / data[0][1] - 1) for i in data],
    label="lev",
    color="tab:blue",
)
# ax7.plot(
#     [i[0] for i in data],
#     [100 * (i[2] / data[0][2] - 1) for i in data],
#     label="l2",
#     color="tab:orange",
# )
# ax7.plot(
#     [i[0] for i in data],
#     [100 * (i[5] / data[0][5] - 1) for i in data],
#     label="L2S",
#     ls="--",
#     color="tab:orange",
# )
ax7.plot(
    [i[0] for i in data],
    [100 * (i[3] / data[0][3] - 1) for i in data],
    label="Mlev",
    color="tab:green",
)
ax7.plot(
    [i[0] for i in data],
    [100 * (i[4] / data[0][4] - 1) for i in data],
    label="MS",
    ls="--",
    color="tab:green",
)
ax7.plot([0, S.swt_done], [0, 0], label="", color="k", linestyle=":")
ax7.legend()
num = len(os.listdir("image"))
plt.savefig("image/" + str(num + 1), dpi=1000)
# plt.subplot(1, 3, 3)
# plt.imshow(S.A, cmap=cmap)
# plt.tight_layout()
# # plt.show()
# plt.savefig("image/" + str(num + 1), dpi=1000)
# result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
# if s_no > 0 and s_no % 10000 == 0:
# print(s_no, 'switches with p =', pos_p)
