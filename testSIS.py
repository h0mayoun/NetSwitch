from SIS import *
from NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle

np.set_printoptions(precision=1, suppress=True)
tmax = 1e9
scale = 1
# G = read_Graph("result/BA-n=1024-k=10-seed=(1,1)/GRDY/7000.mtx")
fig, ax = plt.subplots(1, 2)
Gs = [
    read_Graph("result/BA-n=1024-k=10-seed=(1,1)/SWPC/0.mtx"),
    read_Graph("result/BA-n=1024-k=10-seed=(1,1)/SWPC/16000.mtx"),
    read_Graph("result/BA-n=1024-k=10-seed=(1,1)/GRDY/1100.mtx"),
]
N = Gs[0].shape[0]
lambda1 = np.max(np.real(np.linalg.eigvals(Gs[0])))
k = np.sum(Gs[0], axis=0)

print(1 / lambda1, np.mean(k) / np.mean(k**2))
iterCnt = 10
rho = 0.5
color = ["tab:blue", "tab:orange", "tab:green"]
betaList = np.linspace(0.5, 1.5, 21)
for cnt, G in enumerate(Gs):
    En = []
    NEn = []
    for beta in betaList:
        endemic = np.zeros(iterCnt, dtype=bool)
        Cs = []
        Is = np.zeros(iterCnt)
        lifespan = np.zeros(iterCnt)
        for i in range(iterCnt):
            if i % 20 == 0:
                print(".", end="", flush=True)
            SISmodel = SIS(G, beta / lambda1 * scale, 1 * scale, "hub")
            T, I, C, ls = SISmodel.Gillespie(tmax, samplingRate=1, rho=rho)
            lifespan[i] = ls
            if C[-1] >= N * rho:
                endemic[i] = True
                # ax[2].plot(T, C / N,c = "k",alpha = 0.1,ls = ":")
            # else:
            # ax[1].plot(T, C / N,c = "k",alpha = 0.1,ls = ":")
            # ax[0].plot(T, I / N,c = "k",alpha = 0.1)
            if i > 0 and len(C) < len(Cs[-1]):
                C = np.pad(C, (0, len(Cs[-1]) - len(C)), mode="edge")
            elif i > 0 and len(Cs[-1]) < len(C):
                for j, Csi in enumerate(Cs):
                    Cs[j] = np.pad(Csi, (0, len(C) - len(Csi)), mode="edge")
            Cs.append(C / N)
            Is[i] = np.mean(I) / N
            # if not endemic[i]:
            #    ax[0].scatter(beta, ls, s=1, c="k", alpha=0.1)
            #    ax[1].scatter(beta, C[-1] / N, s=1, c="k", alpha=0.1)
            # else:
            #    ax[0].scatter(beta, ls, s=1, c="r", alpha=0.1)
            #    ax[1].scatter(beta, C[-1] / N, s=1, c="r", alpha=0.1)

        Cs = np.vstack(Cs)
        print()
        print(beta)
        notendemic = np.logical_not(endemic)
        if np.any(endemic == True):
            print(
                "endemic    : mean Lifespan={:.2f}, mean Coverage={:.2f}, mean Infected={:.2f}".format(
                    np.mean(lifespan[endemic]), np.mean(Cs[:, -1], axis=0), np.mean(Is)
                )
            )
            En.append(
                (
                    beta,
                    np.mean(lifespan[endemic]),
                    np.mean(Cs[endemic, -1], axis=0),
                    np.mean(Cs[:, -1], axis=0),
                    np.var(Is[endemic], axis=0) / np.mean(Is[endemic], axis=0),
                )
            )
        else:
            print("endemic    :-------------------")

        if np.any(notendemic == True):
            print(
                "not endemic: mean Lifespan={:.2f}, mean Coverage={:.2f}, mean Infected={:.2f}".format(
                    np.mean(lifespan[notendemic]),
                    np.mean(Cs[:, -1], axis=0),
                    np.mean(Is),
                )
            )
            NEn.append(
                (
                    beta,
                    np.mean(lifespan[notendemic]),
                    np.mean(Cs[notendemic, -1], axis=0),
                    np.mean(Cs[:, -1], axis=0),
                    np.var(Is[notendemic], axis=0) / np.mean(Is[notendemic], axis=0),
                )
            )
        else:
            print("not endemic:-------------------")

    t = [x[0] for x in NEn]
    x1 = [x[1] for x in NEn]
    x2 = [x[4] for x in NEn]
    ax[0].plot(t, x1, color=color[cnt])
    ax[1].plot(t, x2, color=color[cnt])

    t = [x[0] for x in En]
    x1 = [x[1] for x in En]
    x2 = [x[4] for x in En]
    ax[0].plot(t, x1, color=color[cnt], ls=":")
    ax[1].plot(t, x2, color=color[cnt], ls=":")
    # ax[1].plot(t,x2/max(x2),color = color[cnt],ls = ":")

plt.savefig("testSIS.pdf", dpi=300)
