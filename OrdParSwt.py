import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch
import csv
import sys
import matplotlib.animation as animation
import os 
import time

seed = 2
random.seed(seed)
np.set_printoptions(precision = 2,suppress = True,linewidth = np.inf)

def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted ascendingly
    eigVal, eigVec = np.linalg.eig(M)
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]

def getParVec(G,D_n):
    v2, u2 = getLk(I - D_n @ G.A @ D_n, 1)
    s = np.sign(u2)
    return s

def plotAdjacencyImage(G,ax,s):
    n = G.n
    I = np.eye(n)
    
    sPos = (s > 0).astype(np.float32).reshape(-1, 1)
    sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
    img = G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T)
    #sortIdx = np.argsort(s,stable = True)
    #img = img[sortIdx, :][:, sortIdx]
    _ = np.zeros((n,1))
    _[0] = -1
    _[1] = 0
    _[2] = 1
    _[3] = 3
    cmap = colors.ListedColormap(["blue", "white", "green", "red"])
    img = np.hstack((img,_))
    ax.imshow(img,cmap=cmap)
    ax.set_xlim([-0.5,n-0.5])
    ax.set_xticks([])
    ax.set_yticks([])

def plotNetSwitchGraph(G,ax,s,vertex_size = -1,edge_width = 0.1):
    if vertex_size == -1:
        vertex_size = 9600/G.n
        
    color = ["red" if i>0 else "blue" for i in s]
    Gig = ig.Graph.Adjacency(G.A)
    edgecolor = [(0,0.3,0,1) if s[i]!=s[j] else (0,0,0,0.05) for (i,j) in Gig.get_edgelist()]
    edgewidth = [np.log(n)*edge_width if s[i]!=s[j] else np.log(n)*edge_width*0.5 for (i,j) in Gig.get_edgelist()]
    im3 = ig.plot(ig.Graph.Adjacency(G.A), vertex_size=np.log(G.deg)*(vertex_size/np.log(G.deg)[0]),
            edge_width = edgewidth, edge_arrow_size = 0,edge_arrow_width=0,layout="circle",
            target=ax,vertex_color = color,edge_color = edgecolor,vertex_frame_width=0)

n=256
p=np.log2(n)*10/n
kn = 3
graphtype = "ER"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)
elif graphtype == "WS":
    graph = ig.Graph.Watts_Strogatz(dim=1, size=n, nei = kn, p=p)
elif graphtype == "SBM":
    blockSize = np.ones(kn,dtype = np.int64)*np.int64(n/kn)
    blockSize[0] += n - sum(blockSize)
    prefMatrix = np.ones((1,kn))*(1/kn)#np.random.rand(1,kn)
    prefMatrix = prefMatrix.T@prefMatrix
    print(prefMatrix.T@prefMatrix)
    print(blockSize)
    graph = ig.Graph.SBM(n=n, pref_matrix = prefMatrix.tolist(),block_sizes = blockSize.tolist())

G = NetSwitch(graph)

Dinvsqrt = [1.0 / G.deg[i] if i!=0 else 0 for i in G.deg]
D_n = np.diag(Dinvsqrt)
A_n = D_n @ G.A @ D_n
m = sum(G.deg)/2

modularity_limit = np.max(G.base_mod)
#modularity_limit = np.max(G.orthBaseMod)

I = np.eye(n)
v2initial, _ = getLk(D_n @ G.M @ D_n, n-2)
vl2initial, ul2initial = getLk(I - D_n @ G.A @ D_n, 1)
v1initial, _ = getLk(G.A, n-1)
startTime = time.time()
switches = 0
while True:
    print(switches)
    switches+=1
    
    swt = G.modAwareSwitch(modularity_limit,normalized=False)

    if(swt == (-1,-1,-1,-1)):
        break
    
I = np.eye(n)
v2, u2 = getLk(D_n @ G.M @ D_n, n-2)
v1, _ = getLk(G.A, n-1)

vl2final, _ = getLk(I - D_n @ G.A @ D_n, 1)
totalTime = time.time()-startTime
print(str(switches)+" switches in "+str(totalTime)+" (s)")
fig,ax = plt.subplots(1,2,figsize=(10, 5))
if graphtype == "ER":
    st = fig.suptitle("ER; n:"+str(n)+"; p:"+str(np.round(p,2)) +"; seed:"+str(seed)+
                      "\nl2Binitial:" +str(np.round(v2initial,2))+"; l2Bfinal:"+str(np.round(v2,2))+
                      "\nl1Ainitial:" +str(np.round(v1initial,2))+"; l1Afinal:"+str(np.round(v1,2))+
                      "\nl2Linitial:" +str(np.round(vl2initial,2))+"; l2Lfinal:"+str(np.round(vl2final,2))+
                      "\n switches:"+str(switches)+"; time(s):"+str(np.round(totalTime,2)), fontsize="x-large")
elif graphtype == "BA":
    st = fig.suptitle("BA; n:"+str(n)+"; p:"+str(kn) +"; seed:"+str(seed)+
                      "\nl2initial:" +str(np.round(v2initial,2))+"; l2final:"+str(np.round(v2,2))+
                      "\nl1initial:" +str(np.round(v1initial,2))+"; l1final:"+str(np.round(v1,2))+
                      "\nl2Linitial:" +str(np.round(vl2initial,2))+"; l2Lfinal:"+str(np.round(vl2final,2))+
                      "\n switches:"+str(switches)+"; time(s):"+str(np.round(totalTime,2)), fontsize="x-large")
elif graphtype == "WS":
    st = fig.suptitle("WS; n:"+str(n)+"; k:"+str(kn)+"; p:"+str(np.round(p,2)) +"; seed:"+str(seed)+
                      "\nl2initial:" +str(np.round(v2initial,2))+"; l2final:"+str(np.round(v2[1],2))+
                      "\nl1initial:" +str(np.round(v1initial,2))+"; l1final:"+str(np.round(v1,2))+
                      "\nl2Linitial:" +str(np.round(vl2initial,2))+"; l2Lfinal:"+str(np.round(vl2final,2))+
                      "\n switches:"+str(switches)+"; time(s):"+str(np.round(totalTime,2)), fontsize="x-large")
elif graphtype == "SBM":
    st = fig.suptitle("SBM; n:"+str(n)+"; k:"+str(kn) +"; seed:"+str(seed)+
                      "\nl2initial:" +str(np.round(v2initial,2))+"; l2final:"+str(np.round(v2[1],2))+
                      "\nl1initial:" +str(np.round(v1initial,2))+"; l1final:"+str(np.round(v1,2))+
                      "\n switches:"+str(switches)+"; time(s):"+str(np.round(totalTime,2)), fontsize="x-large")
s = getParVec(G,D_n)
plotAdjacencyImage(G,ax[0],s)
plotNetSwitchGraph(G,ax[1],s)  
#ax[2].bar(np.arange(1,G.n+1),G.deg,color = ["red" if i>0 else "blue" for i in s])
#ax[2].set_xlim([0,n])
ax[1].set_aspect('equal', adjustable='box')
num = len(os.listdir("image"))
fig.tight_layout()
plt.savefig("image/"+str(num+1),dpi = 1000)