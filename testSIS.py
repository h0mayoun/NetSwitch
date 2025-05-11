from SIS import *
from NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle

tmax = 500
scale = np.sqrt(1000)
G = read_Graph("result/ER-n=256-p=3.44e-02-seed=(1,1)/GRDY/1387.mtx")
N = G.shape[0]
lambda1 = np.max(np.real(np.linalg.eigvals(G)))
SISmodel = SIS(G, 10 / lambda1 / scale, 1 / scale, "hub")
T, I, C, lifespan = SISmodel.Gillespie(tmax, samplingRate=1)
fig, ax = plt.subplots(1, 2)
ax[0].plot(T, I / N)
ax[1].plot(T, C / N)
plt.savefig("testSIS.pdf", dpi=300)
