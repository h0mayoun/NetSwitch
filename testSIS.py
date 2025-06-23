from SIS import *
from local.NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle
import sys

np.set_printoptions(precision=1, suppress=True)
tmax = 1e9
scale = 100
# G = read_Graph("result/BA-n=1024-k=10-seed=(1,1)/GRDY/7000.mtx")
# 1024 16000 1100
graphDef = sys.argv[1]
id1, id2, id3 = sys.argv[2], sys.argv[3], sys.argv[4]
Gs = [
    read_Graph("result/{}/SWPC/{}.mtx".format(graphDef, id1)),
    read_Graph("result/{}/SWPC/{}.mtx".format(graphDef, id2)),
    read_Graph("result/{}/GRDY/{}.mtx".format(graphDef, id3)),
]
labels = ["ORG", "SWPC", "GRDY"]
# Gs = [
#     read_Graph("result/BA-n=1024-k=10-seed=(1,1)/SWPC/0.mtx"),
# ]
N = Gs[0].shape[0]
lambda1 = np.max(np.real(np.linalg.eigvals(Gs[0])))
k = np.sum(Gs[0], axis=0)

print(1 / lambda1, np.mean(k) / np.mean(k**2))
iterCnt = 100
rho = 0.99
color = ["tab:blue", "tab:orange", "tab:green"]
betaList = np.hstack(
    (np.linspace(0.5, 0.8, 4), np.linspace(0.85, 1.15, 7), np.linspace(1.2, 1.5, 4))
)
savedata = {
    label: {beta: {} for beta in betaList} for label in labels
}  # {beta: {} for beta in betaList}
for cnt, G in enumerate(Gs):
    print(labels[cnt])
    for beta in betaList:
        print("{:.2f}".format(beta), end=" ", flush=True)
        Cs = []
        Is = []
        Ts = []
        lifespan = np.zeros(iterCnt)
        for i in range(iterCnt):
            if i % 50 == 0:
                print(".", end="")
            SISmodel = SIS(
                G, np.sqrt(beta) / lambda1 * scale, 1 * scale / np.sqrt(beta), "hub"
            )
            T, I, C = SISmodel.Gillespie(tmax, samplingRate=0.1, rho=rho)

            # if i > 0 and len(C) < len(Cs[-1]):
            #    C = np.pad(C, (0, len(Cs[-1]) - len(C)), mode="edge")
            # elif i > 0 and len(Cs[-1]) < len(C):
            #    for j, Csi in enumerate(Cs):
            #        Cs[j] = np.pad(Csi, (0, len(C) - len(Csi)), mode="edge")
            Cs.append(C / N)
            Is.append(I / N)
            Ts.append(T)
        # print(endemic,notendemic,lifespan)
        # Cs = np.vstack(Cs)
        # print(beta)
        # print(rhoendemic)
        # print(rcts)
        savedata[labels[cnt]][beta] = {
            "Ts": Ts,
            "Cs": Cs,
            "Is": Is,
        }

with open("result/{}/{}.pkl".format(graphDef, rho), "wb") as f:
    pickle.dump(savedata, f)
