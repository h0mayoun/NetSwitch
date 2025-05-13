from scipy.io import mmread
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.sparse import coo_matrix



n = int((len(sys.argv)-2)/2)


cmap = colors.ListedColormap(["white", "blue"])
fig, ax = plt.subplots(1, n)

for i in range(n):
    A = mmread("result/{}/{}/{}.mtx".format(sys.argv[1], sys.argv[2*(i+1)], sys.argv[2*(i+1)+1]))
    print(max(np.linalg.eigvals(A)))
    ax[i].imshow(A, cmap=cmap)


plt.savefig("compareA.pdf", dpi=1000)
