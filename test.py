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

def calOrdParMod(G):    
    modularity = np.zeros(n)
    B = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            B[i,j] = G.A[i,j] - (G.deg[i]*G.deg[j])/(2*m)

    for p in range(n):
        s = np.array([-1 if i<=p else 1 for i in range(n)])
        modularity[p] = (s.T @ B @ s)/sum(G.deg)
    return modularity

def calculateModularity(G,s):    
    B = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            B[i,j] = G.A[i,j] - (G.deg[i]*G.deg[j])/(2*m)

    return (s.T @ B @ s)/sum(G.deg)

n=64
p=7/n

graph = ig.Graph.Erdos_Renyi(n=n, p=p)
#graph = ig.Graph.Barabasi(n=n, m=3)

G = NetSwitch(graph)
GStd = NetSwitch(graph)

print(G.A)

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

# v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
# s = np.sign(u2)
# sPos = (s > 0).astype(np.float32).reshape(-1, 1)
# sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
swtHeatMap = np.zeros((n,n))
fig = plt.figure(figsize=(12, 6))

cmap = colors.ListedColormap(["tab:blue", "white", "tab:purple", "tab:red"])


# ax1 = fig.add_subplot(3, 3, 1)
# ax1.set_title('Before switching')
# ax1.imshow(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),
#     cmap=cmap)
# ax1.set_xticks([])
# ax1.set_yticks([])


lambdas = np.zeros((6, 0))
swtUpCntSeries = np.zeros((5, 1))
va1, _ = getLk(G.A, n - 1)
vl_2, _ = getLk(I - Dsqrt @ G.A @Dsqrt, 1)
M_ = calculateModularity(G,np.sign(_))

va1Std, _ = getLk(GStd.A, n - 1)
vl_2Std, _ = getLk(I - Dsqrt @ GStd.A@Dsqrt, 1)
MStd_ = calculateModularity(GStd,np.sign(_))

lambdas = np.hstack(
    (
        lambdas,
        np.array(
            [
                [va1],
                [vl_2],
                [M_],
                [va1Std],
                [vl_2Std],
                [MStd_]
            ]
        ),
    )
)


#print(modularity)
checkerboardArea = np.zeros(0)
v20 = vl_2
switches = 0
cnt = 0
while cnt<500:
    cnt+=1
    print(switches,end = " ")
    swt,M = G.modularityAwareSwitch(modularity,modularity_limit)
    print(swt)
    switches +=1
    GStd.switch_A(alg='RAND', count=1)
    if(swt == (-1,-1,-1,-1)):
        break
    
    va1, _ = getLk(G.A, n - 1)
    vl_2, _ = getLk(I - Dsqrt @ G.A @Dsqrt, 1)
    M_ = calculateModularity(G,np.sign(_))

    va1Std, _ = getLk(GStd.A, n - 1)
    vl_2Std, _ = getLk(I - Dsqrt @ GStd.A@Dsqrt, 1)
    MStd_ = calculateModularity(GStd,np.sign(_))

    lambdas = np.hstack(
        (
            lambdas,
            np.array(
                [
                    [va1],
                    [vl_2],
                    [M_],
                    [va1Std],
                    [vl_2Std],
                    [MStd_]
                ]
            ),
        )
    )
            
    i,j,k,l = swt
    swtUpCnt = int(i>k) + int(j>k) + int(i>l) +int(j>l) 
    print(swtUpCnt,j-i,l-k)
    checkerboardArea = np.append(checkerboardArea,(j-i)*(l-k))
    swtUpCntOH = np.zeros((5,1))
    swtUpCntOH[swtUpCnt] = 1
    swtUpCntSeries = np.hstack(
        (
            swtUpCntSeries,swtUpCntOH)
    )
    modularity = M
    print(max(M))
    
print(np.cumsum(swtUpCntSeries,axis = 1)[:,-1])
#print(G.A)
#print(cutlist)
#####################################################################################################################
v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
ax2 = fig.add_subplot(3, 4, 5)
ax2.imshow(G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T),cmap=cmap,
)
ax2.set_xticks([])
ax2.set_yticks([])

ax5 = fig.add_subplot(3, 4, 7)
color = ["red" if i>0 else "blue" for i in s]
Gig = ig.Graph.Adjacency(G.A)
edgecolor = ["black" if s[i]!=s[j] else "gainsboro" for (i,j) in Gig.get_edgelist()]
ig.plot(ig.Graph.Adjacency(G.A), vertex_size=np.log(G.deg)*(5/np.log(G.deg)[0]),edge_width = np.log(n)/5, edge_arrow_size = 0,edge_arrow_width=0,target=ax5,vertex_color = color,edge_color = edgecolor,vertex_frame_width=0)

v2, u2 = getLk(I - Dsqrt @ GStd.A @ Dsqrt, 1)
s = np.sign(u2)
sPos = (s > 0).astype(np.float32).reshape(-1, 1)
sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
color = ["red" if i>0 else "blue" for i in s]
Gig = ig.Graph.Adjacency(GStd.A)
edgecolor = ["black" if s[i]!=s[j] else "gainsboro" for (i,j) in Gig.get_edgelist()]
ax3 = fig.add_subplot(3, 4, 6)
ax3.imshow(GStd.A + np.multiply(GStd.A, 2 * sPos @ sPos.T) - np.multiply(GStd.A, 2 * sNeg @ sNeg.T),cmap=cmap,
)
ax3.set_xticks([])
ax3.set_yticks([])

ax6 = fig.add_subplot(3, 4, 8)
ig.plot(ig.Graph.Adjacency(GStd.A), vertex_size=np.log(GStd.deg)*(5/np.log(GStd.deg)[0]),edge_width = np.log(n)/5, edge_arrow_size = 0,edge_arrow_width=0, target=ax6,vertex_color = color,edge_color = edgecolor,vertex_frame_width=0)

#####################################################################################################################
ax4 = fig.add_subplot(3, 2, 1)
ax4.plot([0,switches],[0,0], label="",color = 'k',linestyle = ':',linewidth = 0.1)
ax4.plot((lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="L2A: a1",color = 'tab:orange')
ax4.plot((lambdas[1, :] - lambdas[1, 0]) / lambdas[1, 0], label="L2A: l2",color = 'tab:orange',linestyle = ':')
ax4.plot((lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="L2A: M",color = 'tab:orange',linestyle = '--')
ax4.plot((lambdas[3, :] - lambdas[3, 0]) / lambdas[3, 0], label="Std: a1",color = 'tab:cyan')
ax4.plot((lambdas[4, :] - lambdas[4, 0]) / lambdas[4, 0], label="Std: l2",color = 'tab:cyan',linestyle = ':')
ax4.plot((lambdas[5, :] - lambdas[5, 0]) / lambdas[5, 0], label="L2A: M",color = 'tab:cyan',linestyle = '--')
#ax4.legend(frameon = False,loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol=4)
ax4.set_xlim([0,switches])



# ax7 = fig.add_subplot(2, 4, 5)
# ax7.plot(calculatmeModularity(G),color = 'tab:orange')
# ax7.plot(initial_modularity,color = 'k')
# ax8 = fig.add_subplot(2, 4, 6)
# ax8.plot(calculatmeModularity(GStd),color = 'tab:cyan')
# ax8.plot(initial_modularity,color = 'k')

ax7 = fig.add_subplot(3, 2, 2)
ax7.plot(calOrdParMod(G),color = 'tab:orange')
ax7.plot(initial_modularity,color = 'k')
ax7.plot(calOrdParMod(GStd),color = 'tab:cyan')
ax7.set_xlim([0,n])

ax8 = fig.add_subplot(3, 2, 5)
swNum = len(checkerboardArea)
window = int(swNum/100)
checkerboardAreaM = [max(checkerboardArea[i:i+window]) for i in range(swNum-window)]
ax8.plot(checkerboardAreaM)
plt.savefig("test-"+str(n)+".pdf",dpi = 1000)