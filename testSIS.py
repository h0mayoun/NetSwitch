from SIS import *
from NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle

np.set_printoptions(precision=1, suppress=True)
tmax = 1e9
scale = 10
# G = read_Graph("result/BA-n=1024-k=10-seed=(1,1)/GRDY/7000.mtx")
Gs = [
    read_Graph("result/BA-n=1024-k=10-seed=(1,1)/SWPC/0.mtx"),
    read_Graph("result/BA-n=1024-k=10-seed=(1,1)/SWPC/16000.mtx"),
    read_Graph("result/BA-n=1024-k=10-seed=(1,1)/GRDY/1100.mtx"),
]
label = ["ORG", "SWPC", "GRDY"]
# Gs = [
#     read_Graph("result/BA-n=1024-k=10-seed=(1,1)/SWPC/0.mtx"),
# ]
N = Gs[0].shape[0]
lambda1 = np.max(np.real(np.linalg.eigvals(Gs[0])))
k = np.sum(Gs[0], axis=0)

print(1 / lambda1, np.mean(k) / np.mean(k**2))
iterCnt = 10
rho = [0.6, 0.8, 1]
fig, ax = plt.subplots(1, 1 + len(rho))
color = ["tab:blue", "tab:orange", "tab:green"]
betaList = np.linspace(0.5, 1.5, 11)
savedata = {beta: {} for beta in betaList}
for cnt, G in enumerate(Gs):
    En = {rhoi: [] for rhoi in rho}
    NEn = []
    for beta in betaList:
        rhoendemic = {rhoi: np.zeros(iterCnt, dtype=bool) for rhoi in rho}
        notendemic = np.zeros(iterCnt, dtype=bool)
        rcts = {rhoi: np.zeros(iterCnt) for rhoi in rho}
        Cs = []
        Is = []
        lifespan = np.zeros(iterCnt)
        for i in range(iterCnt):
            if i % 20 == 0:
                print(".", end="", flush=True)
            SISmodel = SIS(
                G, np.sqrt(beta) / lambda1 * scale, 1 * scale / np.sqrt(beta), "hub"
            )
            T, I, C, ls, rct, rctbool = SISmodel.Gillespie(
                tmax, samplingRate=0.1, rho=rho
            )
            lifespan[i] = ls

            if I[-1] != 0:
                for j, rhoj in enumerate(rho):
                    rhoendemic[rhoj][i] = rctbool[j]
                    rcts[rhoj][i] = rct[j]
            if I[-1] == 0:
                notendemic[i] = True

            if i > 0 and len(C) < len(Cs[-1]):
                C = np.pad(C, (0, len(Cs[-1]) - len(C)), mode="edge")
            elif i > 0 and len(Cs[-1]) < len(C):
                for j, Csi in enumerate(Cs):
                    Cs[j] = np.pad(Csi, (0, len(C) - len(Csi)), mode="edge")
            Cs.append(C / N)
        # print(endemic,notendemic,lifespan)
        Cs = np.vstack(Cs)
        # print(beta)
        # print(rhoendemic)
        # print(rcts)
        savedata[beta] = {
            "rhoendemic": rhoendemic,
            "notendemic": notendemic,
            "rcts": rcts,
            "Cs": Cs,
            "Is": Is,
            "lifespan": lifespan,
        }
        for rhoj in rho:
            if np.any(rhoendemic[rhoj] == True):
                print(
                    "{}-endemic    : mean = {:10.6f}, min = {:10.6f}, max = {:10.6f}, std = {:10.6f}, p = {:10.6f}".format(
                        rhoj,
                        np.mean(rcts[rhoj][rhoendemic[rhoj]]),
                        np.min(rcts[rhoj][rhoendemic[rhoj]]),
                        np.max(rcts[rhoj][rhoendemic[rhoj]]),
                        np.var(rcts[rhoj][rhoendemic[rhoj]]),
                        np.var(rcts[rhoj][rhoendemic[rhoj]])
                        / np.mean(rcts[rhoj][rhoendemic[rhoj]]),
                    )
                )
                En[rhoj].append((beta, np.mean(rcts[rhoj][rhoendemic[rhoj]])))
            else:
                print("{}-endemic    : ###################".format(rhoj))

        if np.any(notendemic == True):
            print(
                "not endemic: mean = {:10.6f}, min = {:10.6f}, max = {:10.6f}, std = {:10.6f}, p = {:10.6f}".format(
                    np.mean(lifespan[notendemic]),
                    np.min(lifespan[notendemic]),
                    np.max(lifespan[notendemic]),
                    np.var(lifespan[notendemic]),
                    np.var(lifespan[notendemic]) / np.mean(lifespan[notendemic]),
                )
            )
            NEn.append(
                (
                    beta,
                    np.var(lifespan[notendemic]) / np.mean(lifespan[notendemic]),
                )
            )
        else:

            print("not endemic : ###################")
    print(En)
    t = [x[0] for x in NEn]
    x1 = [x[1] for x in NEn]
    # x2 = [x[2] for x in NEn]
    ax[0].plot(t, x1, color=color[cnt], label=label[cnt])
    # ax[1].plot(t, x2, color=color[cnt])
    for j, rhoj in enumerate(rho):
        t = [x[0] for x in En[rhoj]]
        x1 = [x[1] for x in En[rhoj]]
        # x2 = [x[2] for x in En]
        ax[j + 1].plot(t, x1, color=color[cnt])


ax[0].legend(frameon=False)
plt.savefig("testSIS.pdf", dpi=300)
