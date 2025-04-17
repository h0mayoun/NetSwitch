import random
import numpy as np
import scipy as sp
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch
import csv
import sys
import matplotlib.animation as animation
import os 
seed = 1
random.seed(seed)
np.set_printoptions(precision = 2,suppress = True,linewidth = np.inf)

def getLk(M, k):
    # get the list of eigenvalue eigenvector sorted ascendingly
    eigVal, eigVec = np.linalg.eig(M)
    #eigval, eigvec = sp.sparse.linalg.eigsh(M,k=2,which = "LM")
    idx = eigVal.argsort()
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    return eigVal[k], eigVec[:, k]


n=128
p=np.log2(n)*1.1/n
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

#modularity_limit = max(G.orthBaseMod)
#modularity_limit = max(G.ordParMod)
I = np.eye(n)
fig = plt.figure(figsize=(9, 9))
cmap = colors.ListedColormap(["tab:blue", "white", "tab:purple", "tab:red"])

lambdas = np.zeros((4, 0))
va1, _ = getLk(G.A, n - 1)
vll2, ul2 = getLk(I - D_n @ G.A @D_n, 1)
vl2, ul2 = getLk(D_n @ G.M @ D_n, n-2)
cut = np.sign(ul2).T@G.A@np.sign(ul2)
modularity_limit = vl2

lambdas = np.hstack(
    (
        lambdas,
        np.array(
            [
                [va1],
                [vll2],
                [vl2],
                [cut]
            ]
        ),
    )
)

switches = 0
ims = []
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 4)
if graphtype == "ER":
    st = fig.suptitle("ER; n="+str(n)+"; p="+str(round(p,2)) +"; seed ="+str(seed), fontsize="x-large")
elif graphtype == "BA":
    st = fig.suptitle("BA; n="+str(n)+"; m="+str(kn) +"; seed ="+str(seed), fontsize="x-large")
#st = fig.suptitle("ER; n="+str(n)+"; p="+str(round(p,2)) +"; seed ="+str(seed), fontsize="x-large")
while True:
    switches+=1
    
    swt = G.modAwareSwitch(modularity_limit,small_switch_limit=0,normalized=False)
    if(swt == (-1,-1,-1,-1)):
        break
    
    
    #v2, u2 = getLk(I - Dsqrt @ G.A @ Dsqrt, 1)
    
    vll2, ul2 = getLk(I - D_n @ G.A @D_n, 1)
    v2, u2 = getLk(D_n @ G.M @ D_n, n-2)
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
    img = np.hstack((img,_))
    im2 = ax2.imshow(img,cmap=cmap)
    ax2.set_xlim([-0.5,n-0.5])
    ax2.set_xticks([])
    ax2.set_yticks([])

    va1, _ = getLk(G.A, n - 1)
    cut = s.T@(np.diag(G.deg)-G.A)@s

    lambdas = np.hstack(
        (
            lambdas,
            np.array(
                [
                    [va1],
                    [vll2],
                    [v2],
                    [cut]
                ]
            ),
        )
    )
    print(switches, cut)
    color = ["red" if i>0 else "blue" for i in s]
    Gig = ig.Graph.Adjacency(G.A)
    #edgecolor = ["black" if s[i]!=s[j] else "gainsboro" for (i,j) in Gig.get_edgelist()]
    edgewidth = [np.log(n)/5 if s[i]!=s[j] else 0 for (i,j) in Gig.get_edgelist()]
    im3 = ig.plot(ig.Graph.Adjacency(G.A), vertex_size=np.log(G.deg)*(50/np.log(G.deg)[0]),
            edge_width = edgewidth, edge_arrow_size = 0,edge_arrow_width=0,layout="circle",
            target=ax3,vertex_color = color,edge_color = "black",vertex_frame_width=0)

    
    im11, = ax1.plot([-1e9,1e9],[0,0], label="",color = 'k',linestyle = ':',linewidth = 0.1)
    im12, = ax1.plot((lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="L1A",color = 'tab:orange')
    im14, = ax1.plot((lambdas[1, :] - lambdas[1, 0]) / lambdas[1, 0], label="L2L",color = 'tab:orange',linestyle = '-.')
    im13, = ax1.plot((lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="L2B",color = 'tab:orange',linestyle = ':')
    im15, = ax1.plot((lambdas[3, :] - lambdas[3, 0]) / lambdas[3, 0], label="Cut",color = 'tab:orange',linestyle = '--')
    # im1, = ax1.plot([0,switches],[0,0], label="",color = 'k',linestyle = ':',linewidth = 0.1,
    #                 (lambdas[0, :] - lambdas[0, 0]) / lambdas[0, 0], label="L2A: a1",color = 'tab:orange',
    #                 (lambdas[1, :] - lambdas[1, 0]) / lambdas[1, 0], label="L2A: l2",color = 'tab:orange',linestyle = ':',
    #                 (lambdas[2, :] - lambdas[2, 0]) / lambdas[2, 0], label="L2A: M",color = 'tab:orange',linestyle = '--')
    if switches == 1:
        ax1.legend()
    #ax4.legend(frameon = False,loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol=4)
    ax1.set_xlim([0,switches])
    #print([im11,im12,im13,im14])
    #artist1, = ax1.get_images()
    #artist2, = ax2.get_images()
    #artist3, = ax3.get_images()
    ims.append([im11,im12,im13,im14,im15,im2,im3])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat=False)
#FFwriter = animation.FFMpegWriter(fps=10)
num = len(os.listdir("animation"))
ani.save("animation/"+"switch"+str(num+1)+".gif",dpi = 300)#,writer=FFwriter)
