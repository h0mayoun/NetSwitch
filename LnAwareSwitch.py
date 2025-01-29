import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch
import sys

np.set_printoptions(precision=2)


def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted ascendingly
    eigVal, eigVec = np.linalg.eig(M)
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]


# Initializing NetSwitch  with an ER network
random.seed(1)
np.random.seed(1)
# option
# 1 add 2 edge to cut
# 2 switch inside one partition
# 3 1 switch in-parition, 1 switch between
# 4 remove 2 edge from cut
option = [1]
n = int(sys.argv[1])
graphtype = sys.argv[2]
if graphtype == "ER":
    p = float(sys.argv[3])
else:
    m = int(sys.argv[3])
vpar = 1
if graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=m)
else:
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
G = NetSwitch(graph)


D = np.diag(np.diag(G.A @ G.A))
Dsqrt = np.diag(1.0 / np.sqrt(np.diag(G.A @ G.A)))
I = np.eye(n)

v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, vpar)
s = np.sign(u2)

sPos = np.astype(s > 0, np.float32).reshape(-1, 1)
sNeg = np.astype(s < 0, np.float32).reshape(-1, 1)
#
plt.figure()
cmap = colors.ListedColormap(["tab:blue", "white", "tab:purple", "tab:red"])
plt.subplot(2, 2, 1)
plt.title("Before switching")
plt.imshow(
    G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
    cmap=cmap,
)
plt.xticks([])
plt.yticks([])

lambdas = np.zeros((5, 0))
diff = np.zeros(0)

# Switch
ite = 1
while True:
    # print(".", end="", flush=True)

    sPos = np.astype(s > 0, np.float32).reshape(-1, 1)
    sNeg = np.astype(s < 0, np.float32).reshape(-1, 1)

    d1 = sPos.T @ D @ sPos
    d2 = sNeg.T @ D @ sNeg
    print("degree sum", min(d1, d2), "-", max(d1, d2))
    print(ite, "\t: before switch:", s @ (D - G.A) @ s / 4, end="\t")
    ite = ite + 1
    spre = s
    cnt = G.switch_A_par_2(s, option)
    print("after switch:", s @ (D - G.A) @ s / 4, end="\t")

    v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, vpar)
    s = np.sign(u2)
    if cnt == -1:
        break

    va1, _ = getLk(G.A, n - 1)
    van123, _ = getLk(I - Dsqrt @ G.A @ Dsqrt, [0, 1, 2])
    vl_2, _ = getLk(D - G.A, 1)

    lambdas = np.hstack(
        (
            lambdas,
            np.array(
                [
                    [abs(va1)],
                    [abs(van123[0])],
                    [abs(van123[1])],
                    [abs(van123[2])],
                    [abs(vl_2)],
                ]
            ),
        )
    )

print("switch done", G.swt_done)
sPos = np.astype(s > 0, np.float32).reshape(-1, 1)
sNeg = np.astype(s < 0, np.float32).reshape(-1, 1)
print(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T))


plt.subplot(2, 2, 2)
plt.title("After switching")
plt.imshow(
    G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
    cmap=cmap,
)
plt.xticks([])
plt.yticks([])
plt.savefig("Switch.png")

plt.subplot(2, 1, 2)
plt.title("Lambda")
plt.plot((lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="a1")
plt.plot((lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="ln2")
plt.plot((lambdas[3, :] - lambdas[3, 0]) / lambdas[3, 0], label="ln3")
plt.plot((lambdas[4, :] - lambdas[4, 0]) / lambdas[4, 0], label="l2")
plt.legend()
# plt.suptitle(graphtype + " n=" + str(n) + "m=" + str(m))
# plt.subplot(3, 1, 3)
# plt.plot(diff, label="diff")
if graphtype == "BA":
    plt.suptitle(
        graphtype
        + " n="
        + str(n)
        + " m="
        + str(m)
        + " option: "
        + str(option)
        + " par vec: "
        + str(vpar + 1)
    )
    plt.savefig(
        "Switch-"
        + graphtype
        + "-"
        + str(n)
        + "-"
        + str(m)
        + "-"
        + str(option)
        + "-"
        + str(vpar + 1)
        + ".pdf",
        dpi=1000,
    )
else:
    plt.suptitle(
        graphtype
        + " n="
        + str(n)
        + " p="
        + str(p)
        + " option: "
        + str(option)
        + " par vec: "
        + str(vpar + 1)
    )
    plt.savefig(
        "Switch-"
        + graphtype
        + "-"
        + str(n)
        + "-"
        + str(p)
        + "-"
        + str(option)
        + "-"
        + str(vpar + 1)
        + ".pdf",
        dpi=1000,
    )
