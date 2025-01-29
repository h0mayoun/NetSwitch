import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch

n = 5
pair = []
for kj in reversed(range(1, n)):
    for j in range(n - kj):
        # print(j + kj, j)
        if kj + j < n and kj + j >= 0:
            pair.append((kj + j, j))

for k, l in pair:
    print(k, l)
# sumpair = [(sum(t), t) for t in pair]
# sort the new list based on the first element (the sum)
# pair = sorted(sumpair, key=lambda x: x[0])
# print(pair)
# print(len(pair))
# # Initializing NetSwitch  with an ER network
# random.seed(1)
# np.random.seed(1)
# n = 256
# p = 0.03
# ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
# G = NetSwitch(ERgraph)

# L = np.array(ERgraph.laplacian())
# A = np.array(ERgraph.get_adjacency().data)
# s = np.array(np.sign(np.random.rand(n) - 0.5)).reshape(-1, 1)

# D = np.diag(np.diag(G.A @ G.A))
# D_half = np.diag(np.sqrt(np.diag(G.A @ G.A)))
# I = np.eye(n)
# # print(D)
# # print("Min Cut:", ERgraph.mincut())

# # Min-cut
# # pid = ERgraph.mincut().partition
# # s[pid[0]] = -1
# # s[pid[1]] = 1

# # eig_values, eig_vectors = np.linalg.eig(D - G.A)
# eig_values, eig_vectors = np.linalg.eig(I - D_half @ G.A @ D_half)
# fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
# fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
# # print(np.sign(fiedler_vector))
# # print(np.sort(eig_values.real))

# s = np.sign(fiedler_vector)
# eig_values, eig_vectors = np.linalg.eig(D - G.A)
# fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
# fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
# print(np.sort(eig_values.real))
# # fiedler

# deltaA = np.zeros((n, n))
# print(s.T @ D @ s - s.T @ G.A @ s)
# print()

# plt.figure()
# cmap = colors.ListedColormap(["white", "tab:blue"])
# plt.subplot(1, 2, 1)
# plt.title("Before switching")
# plt.imshow(G.A, cmap=cmap)
# plt.xticks([])
# plt.yticks([])

# maxtry = 10
# for i in range(10000):
#     prev = s.T @ D @ s - s.T @ G.A @ s
#     cnt = G.switch_A_par(s, alg="RAND", count=1, maxtry=5)
#     eig_values, eig_vectors = np.linalg.eig(I - D_half @ G.A @ D_half)
#     fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
#     fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
#     # print(np.sign(fiedler_vector))
#     # print(i, G.swt_done, np.sort(eig_values.real)[1])
#     # fiedler
#     post = s.T @ D @ s - s.T @ G.A @ s
#     if i % 1000 == 0:
#         print(i, prev, post)
#     s = np.sign(fiedler_vector)

# print("switch done", G.swt_done)

# # print(G.A)
# print(s.T @ D @ s - s.T @ G.A @ s)
# print(D - G.A)

# eig_values, eig_vectors = np.linalg.eig(I - D_half @ G.A @ D_half)
# fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
# fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
# # print(np.sign(fiedler_vector))
# # print(np.sort(eig_values.real)[1])

# eig_values, eig_vectors = np.linalg.eig(D - G.A)
# fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
# fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
# print(np.sort(eig_values.real))

# plt.subplot(1, 2, 2)
# plt.title("After switching")
# plt.imshow(G.A, cmap=cmap)
# plt.xticks([])
# plt.yticks([])
# plt.savefig("Switch.png")
