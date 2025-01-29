import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch

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
n = 512
p = 0.025
m = 5
graphtype = "BA"
if graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=m)
else:
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
G = NetSwitch(graph)


D = np.diag(np.diag(G.A @ G.A))
Dsqrt = np.diag(1.0 / np.sqrt(np.diag(G.A @ G.A)))
I = np.eye(n)


v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)
s = s * s[0]

#
plt.figure()
cmap = colors.ListedColormap(["white", "tab:blue"])
plt.subplot(2, 2, 1)
plt.title("Before switching")
plt.imshow(G.A, cmap=cmap)
plt.xticks([])
plt.yticks([])

lambdas = np.zeros((5, 0))
diff = np.zeros(0)

# Switch
while True:
    # print(".", end="", flush=True)
    # option
    # 1 add 2 edge to cut
    # 2 switch inside one partition
    # 3 1 switch in-parition, 1 switch between
    # 3 remove 2 edge from cut
    print("before switch:", s @ (D - G.A) @ s / 4, end=" ")
    cnt = G.switch_A_par_2(s, [1, 2])
    print("after switch:", s @ (D - G.A) @ s / 4)
    if cnt == -1:
        break
    spre = s

    v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
    s = np.sign(u2)

    diff = np.append(diff, min(abs(sum(s - spre)), abs(sum(s + spre))))

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
plt.subplot(2, 2, 2)
plt.title("After switching")
plt.imshow(G.A, cmap=cmap)
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

# plt.subplot(3, 1, 3)
# plt.plot(diff, label="diff")
if graphtype == "BA":
    plt.savefig("Switch-" + graphtype + "-" + str(n) + "-" + str(m) + "_.pdf", dpi=300)
else:
    plt.savefig("Switch-" + graphtype + "-" + str(n) + "-" + str(p) + "_.pdf", dpi=300)
