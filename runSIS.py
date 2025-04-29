from SIS import *
from NetSwitchAlgsMod import *
from collections import namedtuple
from readGraph import read_Graph
import pickle


def getSS(T, I, mxTime):
    T = np.array(T)
    I = np.array(I)
    if T[-1] >= mxTime:
        return np.mean(I[T > mxTime * 0.9])
    else:
        return 0


file = [
    "email-enron-only.mtx",
    "reptilia-tortoise-network-bsv.edges",
    "inf-USAir97.mtx",
    "aves-wildbird-network.edges",
    "ca-netscience.mtx",
    "ia-radoslaw-email.edges",
    "ca-CSphd.mtx",
    "ca-GrQc.mtx",
]
filenum = 6

G_org = read_Graph("graphs/" + file[filenum])
G_ModA = read_Graph("result/" + file[filenum] + "/ModA-G/13375.mtx")
G_swt = read_Graph("result/" + file[filenum] + "/GRDY/8410.mtx")

# cmap = colors.ListedColormap(['black', 'white'])
# plt.imshow(G_swt) #, cmap=mpl.colormaps['Greys']
# plt.show()

lambda1 = np.max(np.real(np.linalg.eigvals(G_org)))

fig, ax = plt.subplots()
N = G_org.shape[0]


betaCnt = 10
tmax = 100
maxepoch = 10

betaList = np.linspace(0.1, 30, betaCnt)
measurements = {}
lifespan = np.zeros((3, betaCnt, 2))
coverageMean = np.zeros((3, betaCnt, 2))
coverageVar = np.zeros((3, betaCnt, 2))
graphLabels = ["org", "swt", "mod"]
graphs = [G_org, G_ModA, G_swt]
endemicFlag = np.zeros(3)

for i in range(betaCnt):
    print("{}: {:.2f}".format(i, betaList[i] / lambda1 / 10), end=" ", flush=True)
    for j in range(3):
        measurements[(i, j)] = {"lifespan": [], "coverage": [], "infect": []}
        for epoch in range(maxepoch):
            SISmodel = SIS(graphs[j], betaList[i] / lambda1 / 10, 1 / 10, "hub")
            T, I, C = SISmodel.Gillespie(tmax)
            T, I, C = np.array(T), np.array(I), np.array(C)
            measurements[(i, j)]["coverage"].append(C[-1])
            if T[-1] <= tmax:
                measurements[(i, j)]["lifespan"].append(T[-1])
                measurements[(i, j)]["infect"].append(0)
            else:
                measurements[(i, j)]["lifespan"].append(np.inf)
                measurements[(i, j)]["infect"].append(np.mean(I[T >= tmax * 0.9]))
            print(".", end="", flush=True)
    print("")

color = ["r", "g", "b"]
for i in range(betaCnt):
    for j in range(3):
        ax.scatter(
            betaList[i] / lambda1,
            np.mean(measurements[(i, j)]["coverage"]) / N,
            color=color[j],
        )
    # idx = lifespan[j,:,1] > maxepoch*admitfrac
    # ax.plot(betaList[idx]/lambda1,lifespan[j,idx,0]/lifespan[j,idx,1],label = graphLabels[j])

ax.legend()
fig.savefig(file[filenum] + ".pdf")

with open("result/" + file[filenum] + "/data.pkl", "wb") as f:
    pickle.dump(measurements, f)
# mxTime = 10000

# for r0 in r0List:
#     Ilog = np.zeros((4,0))
#     for i in range(100):
#         mu = beta/r0
#         print(i,end = " ",flush=True)
#         infected = int(np.floor(i0 * N))
#         infected = np.random.permutation([1] * infected + [0] * (N - infected)).tolist()
#         SISmodelSwt = SIS(G_swt, beta, mu, infected)
#         SISmodelSwt.step_simulation(mxTime)
#         SwtICnt = getSS(SISmodelSwt.Ts,SISmodelSwt.Is,mxTime)
#         SISmodelModA = SIS(G_ModA, beta, mu, infected)
#         SISmodelModA.step_simulation(mxTime)
#         ModAICnt = getSS(SISmodelModA.Ts,SISmodelModA.Is,mxTime)
#         SISmodelOrg = SIS(G_org, beta, mu, infected)
#         SISmodelOrg.step_simulation(mxTime)
#         OrgICnt = getSS(SISmodelOrg.Ts,SISmodelOrg.Is,mxTime)

#         Ilog = np.hstack((Ilog,np.array([[beta/mu],[SwtICnt],[ModAICnt],[OrgICnt]])))
#     mn = np.mean(Ilog,axis = 1)
#     print("\n",beta/mu,mn)
#     ax.scatter([beta/mu],[mn[1]/N],color = "r")
#     ax.scatter([beta/mu],[mn[2]/N],color = "g")
#     ax.scatter([beta/mu],[mn[3]/N],color = "b")

# fig.savefig(file[filenum] + ".pdf")
# for i in range(100):
#     SISmodel.step_simulation()
#     print(SISmodel.count_infections())
# SISmodel.plot_epi_ts(file[filenum] + ".pdf")
# print(SISmodel.count_infections())
# SISmodel.animate_spread(1, 1000, frame_duration=0.1, gif_name=file[filenum])
# SISmodel.plot_epi_ts(file[filenum] + "org.pdf")
