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

seed = 1
random.seed(seed)
np.set_printoptions(precision = 2,suppress = True,linewidth = np.inf)

def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted ascendingly
    eigVal, eigVec = np.linalg.eig(M)
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]

def plotAdjacencyImage(G,ax):
    n = G.n
    I = np.eye(n)
    
    v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
    s = np.sign(u2)
    s = s*s[0]
    sPos = (s > 0).astype(np.float32).reshape(-1, 1)
    sNeg = (s < 0).astype(np.float32).reshape(-1, 1)
    img = G.A + np.multiply(G.A, 2 * sPos @ sPos.T) - np.multiply(G.A, 2 * sNeg @ sNeg.T)
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


n=128
p=0.1#10/n
kn = 4
graphtype = "ER"
if graphtype == "ER":
    graph = ig.Graph.Erdos_Renyi(n=n, p=p)
elif graphtype == "BA":
    graph = ig.Graph.Barabasi(n=n, m=kn)

G = NetSwitch(graph)
Dsqrt = np.diag(1.0 / np.sqrt(G.deg))
A_n = Dsqrt @ G.A @ Dsqrt
m = sum(G.deg)/2

modularity_limit = max(G.ordParMod)
I = np.eye(n)
v2initial, _ = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
v1initial, _ = getLk(G.A, n-1)

startTime = time.time()
switches = 0

basis = G.orthBase
while True:
    #print(switches)
    switches+=1
    
    swt = G.modAwareSwitch(modularity_limit,small_switch_limit=n/np.log(n))

    if(swt == (-1,-1,-1,-1)):
        break
    
I = np.eye(n)
v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, range(n))
v1, _ = getLk(G.A, n-1)

totalTime = time.time()-startTime
print(str(switches)+" switches in "+str(totalTime)+" (s)")
fig,ax = plt.subplots(figsize=(5, 5))
if graphtype == "ER":
    st = fig.suptitle("ER; n:"+str(n)+"; p:"+str(round(p,2)) +"; seed:"+str(seed)+
                      "\nl2initial:" +str(round(v2initial,2))+"; l2final:"+str(round(v2[1],2))+
                      "\nl1initial:" +str(round(v1initial,2))+"; l1final:"+str(round(v1,2))+
                      "\n switches:"+str(switches)+"; time(s):"+str(round(totalTime,2)), fontsize="x-large")
elif graphtype == "BA":
    st = fig.suptitle("BA; n:"+str(n)+"; p:"+str(round(p,2)) +"; seed:"+str(seed)+
                      "\nl2initial:" +str(round(v2initial,2))+"; l2final:"+str(round(v2[1],2))+
                      "\nl1initial:" +str(round(v1initial,2))+"; l1final:"+str(round(v1,2))+
                      "\n switches:"+str(switches)+"; time(s):"+str(round(totalTime,2)), fontsize="x-large")
plotAdjacencyImage(G,ax)   
num = len(os.listdir("image"))
fig.tight_layout()
plt.savefig("image/"+str(num+1),dpi = 1000)