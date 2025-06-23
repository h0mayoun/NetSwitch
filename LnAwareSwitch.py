import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from local.NetSwitchAlgsMod import NetSwitch
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
# option
# 1 add 2 edge to cut
# 2 switch inside partition
# 3 1 switch in-parition, 1 switch between
# 4 remove 2 edge from cut
option = [1,3]
n = int(sys.argv[1])
graphtype = sys.argv[2]
if graphtype == "ER":
    p = float(sys.argv[3])
else:
    m = int(sys.argv[3])

seed = int(sys.argv[4])
random.seed(seed)
np.random.seed(seed)
vpar = 1
if graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=m)
else:
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
G = NetSwitch(graph)
GStd = NetSwitch(graph)

D = np.diag(np.diag(G.A @ G.A))
Dsqrt = np.diag(1.0 / np.sqrt(np.diag(G.A @ G.A)))
I = np.eye(n)

v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)

sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
#
plt.figure()
cmap = colors.ListedColormap(["tab:blue", "white", "tab:purple", "tab:red"])
plt.subplot(2, 3, 1)
plt.title("Before switching")
plt.imshow(
    G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
    cmap=cmap,
)
plt.xticks([])
plt.yticks([])

lambdas = np.zeros((4, 0))
diff = np.zeros(0)

va1, _ = getLk(G.A, n - 1)
vl_2, _ = getLk(D - G.A, 1)

va1Std, _ = getLk(GStd.A, n - 1)
vl_2Std, _ = getLk(D - GStd.A, 1)
lambdas = np.hstack(
    (
        lambdas,
        np.array(
            [
                [abs(va1)],
                [abs(vl_2)],
                [abs(va1Std)],
                [abs(vl_2Std)]
            ]
        ),
    )
)

ref2 = s @ (D - G.A) @ s / 4
ref1 = s @ (D - G.A) @ s / 8
p2 = 1 + ref2
p1 = 1/np.log(p2-ref1)    


# Switch
ite = 1
while True:
    # print(".", end="", flush=True)

    sPos = (s > 0).astype(np.float32).reshape(-1, 1)
    sNeg = (s < 0).astype(np.float32).reshape(-1, 1)

    d1 = sPos.T @ D @ sPos
    d2 = sNeg.T @ D @ sNeg
    print("degree sum", min(d1, d2), "-", max(d1, d2))
    if s @ (D - G.A) @ s / 4 >= ref2:
        ref2 = s @ (D - G.A) @ s / 4
        ref1 = s @ (D - G.A) @ s / 8
        p2 = 1 + ref2
        p1 = 1/np.log(p2-ref1)    
        temp = 1.0
    else:
        temp = 1-p1*np.log(p2-s @ (D - G.A) @ s / 4)
    chance = random.random()
    print(ite, "\t: before switch:", s @ (D - G.A) @ s / 4, "\t temp",round(temp,2),"\tchance ",round(chance,2), end="\t")
    ite = ite + 1
    spre = s
    chance = random.random()
    cnt,swtype = G.switch_A_par(s, option,alg='RAND',maxtry = 5000,tempCheck = temp>=chance)
    print("\ttype",swtype, end="\t")
    v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, vpar)
    s = np.sign(u2)
    print("after switch:", s @ (D - G.A) @ s / 4, "\t temp",round(temp,2), end="\t")
    if cnt == -1:
        break

    GStd.switch_A(alg='RAND', count=1)
    
    va1, _ = getLk(G.A, n - 1)
    vl_2, _ = getLk(D - G.A, 1)
    
    va1Std, _ = getLk(GStd.A, n - 1)
    vl_2Std, _ = getLk(D - GStd.A, 1)
    lambdas = np.hstack(
        (
            lambdas,
            np.array(
                [
                    [abs(va1)],
                    [abs(vl_2)],
                    [abs(va1Std)],
                    [abs(vl_2Std)]
                ]
            ),
        )
    )
    

print("switch done", G.swt_done)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
print(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T))


plt.subplot(2, 3, 2)
plt.title("After L2A switching")
plt.imshow(
    G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
    cmap=cmap,
)
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 3)
plt.title("After Std switching")
plt.imshow(
    GStd.A + np.multiply(GStd.A, 2 * sPos @ sPos.T) - np.multiply(GStd.A, 2 * sNeg @ sNeg.T),
    cmap=cmap,
)
plt.xticks([])
plt.yticks([])
#plt.savefig("Switch.png")

plt.subplot(2, 1, 2)
plt.title("Lambda")
plt.plot((lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="L2A: a1")
plt.plot((lambdas[1, :] - lambdas[1, 0]) / lambdas[1, 0], label="L2A: l2")
plt.plot((lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="Std: a1")
plt.plot((lambdas[3, :] - lambdas[3, 0]) / lambdas[3, 0], label="Std: l2")
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
        + str(seed)
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
        + str(seed)
        + ".pdf",
        dpi=1000,
    )
