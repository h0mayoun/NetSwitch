from NetSwitchAlgsMod import *
import pickle
import random
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import copy
import os
from scipy.io import mmread
from readGraph import read_Graph
np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)
np.random.seed(1)
random.seed(1)

n = 64
p = np.log2(n) * 1.1 / n
kn = 3
graphtype = "ER"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)
S = NetSwitch(graph)

# A = read_Graph("graphs/email-enron-only.mtx")
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
# degVec = S.deg.reshape(1,-1)
# for u in range(1,S.n):
#     s = np.array([-S.deg[i]/np.sqrt(2*S.m*S.n) if i < u else S.deg[i]/np.sqrt(2*S.m*S.n) for i in range(S.n)]).reshape(1,-1)
#     #print(s)
#     modAprx[u] = np.mean(degVec)-np.sum(s.T@s)-4*max([0,max(S.deg)-u+1])/n
# print(modAprx[0:10])
# print(S.deg[0:10])
# S.switch_A(alg="GRDY")
# S.plotAdjacencyImage(ax[1,0])
# S.plotNetSwitchGraph(ax[1,1])
# ax[1,1].axis('equal')
# ax[1,2].plot(S.base_mod)
# ax[1,2].plot(modAprx)

# plt.savefig("test.png", dpi=1000)
fig = plt.figure(figsize=(9, 9))
ax1,ax2,ax3 = fig.add_subplot(3,3,1),fig.add_subplot(3,3,3),fig.add_subplot(3,3,4)
S.plotAdjacencyImage(ax1)
S.plotNetSwitchGraph(ax2)
ax3.plot(S.base_mod)
ax2.axis('equal')



data = [(S.swt_done, S.lev(), S.l2(normed=True), S.Mlev(normed=False), S.MScore(normed=False))]
while True:
    swt_num = S.switch_A(alg="SWOP", count=1)
    data.append((S.swt_done, S.lev(), S.l2(normed=True), S.Mlev(normed=False), S.MScore(normed=False)))
    print(S.swt_done)
    if swt_num != -1:
        break

print("did ",(S.swt_done)," switches")
print("rejected ",(S.swt_rejected)," switches")

ax4,ax5,ax6 = fig.add_subplot(3,3,2),fig.add_subplot(3,3,6),fig.add_subplot(3,3,5)
S.plotAdjacencyImage(ax4)
S.plotNetSwitchGraph(ax5)
ax6.plot(S.base_mod)
ax5.axis('equal')



ax7 = fig.add_subplot(3,1,3)
ax7.plot([i[0] for i in data], [100 * (i[1] / data[0][1] - 1) for i in data],label = "lev")
ax7.plot([i[0] for i in data], [100 * (i[2] / data[0][2] - 1) for i in data],label = "l2")
ax7.plot([i[0] for i in data], [100 * (i[3] / data[0][3] - 1) for i in data],label = "Mlev")
ax7.plot([i[0] for i in data], [100 * (i[4] / data[0][4] - 1) for i in data],label = "MS")
ax7.plot([0,S.swt_done],[0,0],label = "",color="k",linestyle = ":")
ax7.legend()
num = len(os.listdir("image"))
plt.savefig("image/" + str(num + 1), dpi=1000)
# plt.subplot(1, 3, 3)
# plt.imshow(S.A, cmap=cmap)
# plt.tight_layout()
# # plt.show()
#plt.savefig("image/" + str(num + 1), dpi=1000)
#result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
#if s_no > 0 and s_no % 10000 == 0:
# print(s_no, 'switches with p =', pos_p)
