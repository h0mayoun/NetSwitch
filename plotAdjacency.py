from scipy.io import mmread
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.sparse import coo_matrix

fig, ax = plt.subplots(1, 2)
print(sys.argv)
A = mmread("result/{}/{}/{}.mtx".format(sys.argv[1], sys.argv[2], sys.argv[3]))
cmap = colors.ListedColormap(["white", "blue"])
ax[0].imshow(A, cmap=cmap)

A = mmread("result/{}/{}/{}.mtx".format(sys.argv[1], sys.argv[4], sys.argv[5]))
ax[1].imshow(A, cmap=cmap)

plt.savefig("compareA.pdf", dpi=1000)
