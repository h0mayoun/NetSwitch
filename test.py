from NetSwitchAlgsMod2 import *
import pickle
import random
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import copy
import os
from scipy.io import mmread

np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)
np.random.seed(1)
random.seed(1)

# n = 64
# p = np.log2(n) * 1.05 / n
# random.seed(0)
# ERgraph = ig.Graph.Erdos_Renyi(n=n, p=p)
enronG = mmread("email-enron-only.mtx")
S = NetSwitch(ig.Graph.Adjacency(enronG.todense()))
Aorg = copy.copy(S.A)
data = [(S.swt_done, S.lev(), S.l2(normed=True), S.Mlev(normed=False))]
while True:
    swt_num = S.switch_A(alg="SWOP", count=1)
    data.append((S.swt_done, S.lev(), S.l2(normed=True), S.Mlev(normed=False)))
    print(S.swt_done)
    if swt_num != -1:
        break

print(S.swt_rejected)
print(S.swt_done)

cmap = colors.ListedColormap(["white", "tab:blue"])
plt.figure(figsize=(15, 3))
plt.subplot(1, 3, 1)
plt.imshow(Aorg, cmap=cmap)
plt.subplot(1, 3, 2)
plt.plot([i[0] for i in data], [100 * (i[1] / data[0][1] - 1) for i in data])
plt.plot([i[0] for i in data], [100 * (i[2] / data[0][2] - 1) for i in data])
plt.plot([i[0] for i in data], [100 * (i[3] / data[0][3] - 1) for i in data])
plt.subplot(1, 3, 3)
plt.imshow(S.A, cmap=cmap)
plt.tight_layout()
# plt.show()
num = len(os.listdir("image"))
plt.savefig("image/" + str(num + 1), dpi=1000)
# result[pos_p].append((S.swt_done, S.assortativity_coeff(), S.total_checkers()))
# if s_no > 0 and s_no % 10000 == 0:
#  print(s_no, 'switches with p =', pos_p)
