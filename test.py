import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch
np.set_printoptions(precision = 2,suppress = True,linewidth = np.inf)
import csv
import sys
random.seed(1)
def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted ascendingly
    eigVal, eigVec = np.linalg.eig(M)
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]

def calculatmeModularity(G):    
    modularity = np.zeros(n)
    B = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            B[i,j] = G.A[i,j] - (G.deg[i]*G.deg[j])/(2*m)

    for p in range(n):
        s = np.array([-1 if i<=p else 1 for i in range(n)])
        modularity[p] = (s.T @ B @ s)/sum(G.deg)
    return modularity

n=64
p=7/n

graph = ig.Graph.Erdos_Renyi(n=n, p=p)
#graph = ig.Graph.Barabasi(n=n, m=3)

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
initial_modularity = modularity
modularity_limit = max(modularity)
I = np.eye(n)

v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)

fig = plt.figure(figsize=(12, 6))

cmap = colors.ListedColormap(["tab:blue", "white", "tab:purple", "tab:red"])


# ax1 = fig.add_subplot(3, 3, 1)
# ax1.set_title('Before switching')
# ax1.imshow(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
#     cmap=cmap)
# ax1.set_xticks([])
# ax1.set_yticks([])


l2,_ = getLk(np.diag(G.deg)-G.A,1)
D = np.diag(G.deg)
va1, _ = getLk(G.A, n - 1)
vl_2, _ = getLk(I - Dsqrt @ G.A @Dsqrt, 1)
va1Std, _ = getLk(GStd.A, n - 1)
vl_2Std, _ = getLk(I - Dsqrt @ GStd.A@Dsqrt, 1)
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
switches = 0
while True:
    print(switches,end = " ")
    sw,M = G.switch_A_3(modularity,modularity_limit,count = 1,alg='RAND')
    switches +=1
    GStd.switch_A(alg='RAND', count=1)
    if(sw != -1):
        break
    
    va1, _ = getLk(G.A, n - 1)
    vl_2, _ = getLk(I - Dsqrt @ G.A @Dsqrt, 1)
    va1Std, _ = getLk(GStd.A, n - 1)
    vl_2Std, _ = getLk(I - Dsqrt @ GStd.A@Dsqrt, 1)
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
    

#print(G.A)
#print(cutlist)
v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)

#####################################################################################################################
ax2 = fig.add_subplot(2, 4, 5)
ax2.imshow(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),cmap=cmap,
)
ax2.set_xticks([])
ax2.set_yticks([])

v2, u2 = getLk(I - Dsqrt @ GStd.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)

ax3 = fig.add_subplot(2, 4, 6)
ax3.imshow(GStd.A + np.multiply(GStd.A, 2 * sPos @ sPos.T) - np.multiply(GStd.A, 2 * sNeg @ sNeg.T),cmap=cmap,
)
ax3.set_xticks([])
ax3.set_yticks([])

#####################################################################################################################
ax4 = fig.add_subplot(2, 2, 1)
ax4.plot([0,switches],[0,0], label="",color = 'k',linestyle = ':',linewidth = 0.1)
ax4.plot((lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="L2A: a1",color = 'tab:orange')
ax4.plot((lambdas[1, :] - lambdas[1, 0]) / lambdas[1, 0], label="L2A: l2",color = 'tab:orange',linestyle = ':')
ax4.plot((lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="Std: a1",color = 'tab:cyan')
ax4.plot((lambdas[3, :] - lambdas[3, 0]) / lambdas[3, 0], label="Std: l2",color = 'tab:cyan',linestyle = ':')
#ax4.legend(frameon = False,loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol=4)
ax4.set_xlim([0,switches])

ax5 = fig.add_subplot(2, 4, 7)
ig.plot(ig.Graph.Adjacency(G.A), vertex_size=np.log(G.deg)*(5/np.log(G.deg)[0]),edge_width = 0.5/n, edge_arrow_size = 0,edge_arrow_width=0,target=ax5)
ax6 = fig.add_subplot(2, 4, 8)
ig.plot(ig.Graph.Adjacency(GStd.A), vertex_size=np.log(GStd.deg)*(5/np.log(GStd.deg)[0]),edge_width = 0.5/n, edge_arrow_size = 0,edge_arrow_width=0, target=ax6)

# ax7 = fig.add_subplot(2, 4, 5)
# ax7.plot(calculatmeModularity(G),color = 'tab:orange')
# ax7.plot(initial_modularity,color = 'k')
# ax8 = fig.add_subplot(2, 4, 6)
# ax8.plot(calculatmeModularity(GStd),color = 'tab:cyan')
# ax8.plot(initial_modularity,color = 'k')

ax7 = fig.add_subplot(2, 2, 2)
ax7.plot(calculatmeModularity(G),color = 'tab:orange')
ax7.plot(initial_modularity,color = 'k')
ax7.plot(calculatmeModularity(GStd),color = 'tab:cyan')
ax7.set_xlim([0,n])

plt.savefig("test-"+str(n)+".pdf",dpi = 1000)