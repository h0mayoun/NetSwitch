from SIS import *
from NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle
np.set_printoptions(precision=1,suppress =True)
tmax = 1000000
scale = np.sqrt(1)
#G = read_Graph("result/BA-n=1024-k=10-seed=(1,1)/GRDY/7000.mtx")
fig, ax = plt.subplots(1, 2)
Gs = [read_Graph("result/BA-n=256-k=8-seed=(1,1)/SWPC/0.mtx"),
      read_Graph("result/BA-n=256-k=8-seed=(1,1)/SWPC/5009.mtx"),
      read_Graph("result/BA-n=256-k=8-seed=(1,1)/GRDY/300.mtx")]
N = Gs[0].shape[0]
lambda1 = np.max(np.real(np.linalg.eigvals(Gs[0])))
iterCnt = 100
rho = 1
color = ["tab:blue","tab:orange","tab:green"]
betaList = np.linspace(1e-9, 2, 21)
for cnt,G in enumerate(Gs):
    En = []
    NEn = []
    for beta in betaList:
        endemic = np.zeros(iterCnt, dtype=bool)
        Cs = []
        lifespan = np.zeros(iterCnt)
        for i in range(iterCnt):
            if i%20 == 0:
                print(".",end = "",flush=True)
            SISmodel = SIS(G, beta / lambda1 / scale, 1 / scale, "hub")
            T, I, C, ls = SISmodel.Gillespie(tmax, samplingRate=1,rho = rho)
            lifespan[i] = ls
            if C[-1]>=N*rho:
                endemic[i] = True
                #ax[2].plot(T, C / N,c = "k",alpha = 0.1,ls = ":")
            #else:
                #ax[1].plot(T, C / N,c = "k",alpha = 0.1,ls = ":")
            #ax[0].plot(T, I / N,c = "k",alpha = 0.1)
            if i>0 and len(C)<len(Cs[-1]):
                C = np.pad(C, (0, len(Cs[-1])-len(C)), mode='edge')
            elif i>0 and len(Cs[-1])<len(C):
                for j,Csi in enumerate(Cs):
                    Cs[j] = np.pad(Csi, (0, len(C)-len(Csi)), mode='edge')
            Cs.append(C/N)

        Cs = np.vstack(Cs)
        print()
        print(beta)
        notendemic = np.logical_not(endemic)
        if np.any(endemic == True):
            #ax[2].plot(np.mean(Cs[endemic],axis = 0),c = "k")
            print("endemic    : mean={:.2f}, variance={:.2f}".format(np.mean(lifespan[endemic]),np.var(lifespan[endemic])))
            En.append((beta,np.mean(lifespan[endemic]),np.var(lifespan[endemic])))
        else:
            print("endemic    :-------------------")
        if np.any(notendemic == True):
            #ax[1].plot(np.mean(Cs[notendemic],axis = 0),c = "k")
            print("not endemic: mean={:.2f}, variance={:.2f}".format(np.mean(lifespan[notendemic]),np.var(lifespan[notendemic])))
            NEn.append((beta,np.mean(lifespan[notendemic]),np.var(lifespan[notendemic])))
        else:
            print("not endemic:-------------------")
       
  

    t = [x[0] for x in NEn]
    x1 = [x[1] for x in NEn]
    x2 = [x[2] for x in NEn]
    ax[0].plot(t,x1,color = color[cnt])  
    #ax[0].plot(t,x2/max(x2),color = color[cnt],ls = ":")
    
    t = [x[0] for x in En]
    x1 = [x[1] for x in En]
    x2 = [x[2] for x in En]
    ax[1].plot(t,x1,color = color[cnt])  
    #ax[1].plot(t,x2/max(x2),color = color[cnt],ls = ":")

plt.savefig("testSIS.pdf", dpi=300)
