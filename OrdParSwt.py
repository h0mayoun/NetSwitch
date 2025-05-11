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

n = 64
p = np.log2(n) * 1.2 / n
kn = 3
graphtype = "ER"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    graph_des = "ER-n={}-p={:.2e}-seed=({},{})".format(n, p, seed1, seed2)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)
    graph_des = "BA-n={}-k={}-seed=({},{})".format(n, kn, seed1, seed2)
S = NetSwitch(graph)

# filenum = 4
# A = read_Graph("graphs/" + file[filenum])
# n = A.shape[0]
# graph_des = file[filenum]
# deg = sorted(np.sum(A, axis=1), reverse=True)
# base = np.zeros((n, 0))
# S = NetSwitch(ig.Graph.Adjacency(A))
# print(deg)
# for u in range(n+1):
#     s = np.ones((n,1))

#     e = n
#     b = n-u
#     while b and u > deg[e-1]:
#         b -=1
#         e -=1
#     s[b:e] = s[b:e]*-1
#     base = np.hstack((base,s))
# S = NetSwitch(ig.Graph.Adjacency(A),base = base)
#


print(S.assortativity_coeff())
fig = plt.figure(figsize=(9, 9))
plt.suptitle(graph_des)
ax1, ax2, ax3 = (
    fig.add_subplot(3, 3, 1),
    fig.add_subplot(3, 3, 3),
    fig.add_subplot(3, 3, 4),
)
print("1")
S.plotAdjacencyImage(ax1)
S.plotNetSwitchGraph(ax2)
ax3.plot(S.base_mod)
ax2.axis("equal")

print("2")

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
if sys.argv[1] == "1" and not os.path.exists("result/{}/{}/".format(graph_des, alg)):
    os.makedirs("result/{}/{}/".format(graph_des, alg))
    mmwrite("result/{}/{}/{}.mtx".format(graph_des, alg, S.swt_done), S.A)
while True:
    # print(S.MScore(normed=False))
    swt_num = S.modularitySwitch(count=10)
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
    if sys.argv[1] == "1":
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
ax6.set_xlim([0, n])
# ax6.plot(S.M_lb)
ax5.axis("equal")


ax7 = fig.add_subplot(3, 1, 3)
ax7.plot(
    [i[0] for i in data],
    [100 * (i[1] / data[0][1] - 1) for i in data],
    label="lev",
    color="tab:blue",
)
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
    ls="-.",
    color="tab:green",
)
ax7.plot([0, S.swt_done], [0, 0], label="", color="k", linestyle=":")
ax7.legend()
num = len(os.listdir("image"))
plt.savefig("image/" + str(num + 1) + ".pdf", dpi=1000)
# plt.subplot(1, 3, 3)
# plt.imshow(S.A, cmap=cmap)
# plt.tight_layout()
# # plt.show()
# plt.savefig("image/" + str(num + 1), dpi=1000)
# result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
# if s_no > 0 and s_no % 10000 == 0:
# print(s_no, 'switches with p =', pos_p)
