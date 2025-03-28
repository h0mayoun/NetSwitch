import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch
np.set_printoptions(precision = 2,suppress = True,linewidth = np.inf)
import csv
import sys
#random.seed(3)
def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted ascendingly
    eigVal, eigVec = np.linalg.eig(M)
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]

n=128
p=12/n

graph = ig.Graph.Barabasi(n=n, m=4)

G = NetSwitch(graph)
GStd = NetSwitch(graph)

print(G.A)
lambdas = np.zeros((4, 0))

cutlist = np.zeros(n)
for i in range(n):
    for j in range(i+1,n):
        if G.A[i,j] != 0:
            for k in range(j+1):
                if k>=i:
                    cutlist[k] +=1
                

#print(G.A)
#print(cutlist) 
Dsqrt = np.diag(1.0 / np.sqrt(G.deg))
A_n = Dsqrt @ G.A @ Dsqrt
m = sum(G.deg)/2

modularity = np.zeros(n)
B = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        B[i,j] = G.A[i,j] - (G.deg[i]*G.deg[j])/(2*m)

for p in range(n):
    s = np.array([-1 if i<=p else 1 for i in range(n)])
    modularity[p] = (s.T @ B @ s)/sum(G.deg)

cumsumdeg = np.cumsum(G.deg)
expected = cutlist#[cutlist[i] if cutlist[i]<cumsumdeg[i] else cumsumdeg[i] for i in range(n)]
#print(expected)

I = np.eye(n)

v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)

plt.figure()
cmap = colors.ListedColormap(["tab:blue", "white", "tab:purple", "tab:red"])
plt.subplot(2, 3, 1)
plt.title('Before switching')
plt.imshow(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
    cmap=cmap)
plt.xticks([])
plt.yticks([])


l2,_ = getLk(np.diag(G.deg)-G.A,1)
D = np.diag(G.deg)
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


#print(modularity)
v20 = vl_2
while True:
    sw,M = G.switch_A_3(modularity,count = 1,alg='RAND')
    GStd.switch_A(alg='RAND', count=1)
    if(sw != -1):
        break
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
    
    modularity = M
    print(max(M))
    

modularity = np.zeros(n)
B = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        B[i,j] = G.A[i,j] - (G.deg[i]*G.deg[j])/(2*m)

for p in range(n):
    s = np.array([-1 if i<=p else 1 for i in range(n)])
    modularity[p] = s.T @ B @ s

print(max(modularity))

#print(G.A)
#print(cutlist)
v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
plt.subplot(2, 3, 2)
plt.title("After L2A switching")
plt.imshow(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),cmap=cmap,
)
plt.xticks([])
plt.yticks([])

v2, u2 = getLk(I - Dsqrt @ GStd.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
plt.subplot(2, 3, 3)
plt.title("After Std switching")
plt.imshow(GStd.A + np.multiply(GStd.A, 2 * sPos @ sPos.T) - np.multiply(GStd.A, 2 * sNeg @ sNeg.T),cmap=cmap,
)
#(GStd.A + np.multiply(GStd.A, 2 * sPos @ sPos.T) - np.multiply(GStd.A, 2 * sNeg @ sNeg.T))
plt.xticks([])
plt.yticks([])

l2,_ = getLk(np.diag(G.deg)-G.A,1)
#print(l2)

plt.subplot(2, 1, 2)
plt.title("Lambda")
plt.plot((lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="L2A: a1")
plt.plot((lambdas[1, :] - lambdas[1, 0]) / lambdas[1, 0], label="L2A: l2")
plt.plot((lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="Std: a1")
plt.plot((lambdas[3, :] - lambdas[3, 0]) / lambdas[3, 0], label="Std: l2")
plt.legend()

plt.savefig("test-"+str(n)+".pdf",dpi = 1000)