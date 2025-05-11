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
filenum = 4

# graph_des = "BA-n=128-k=3-seed=(1,1)"
# files = [
#     "result/BA-n=128-k=3-seed=(1,1)/GRDY/0.mtx",
#     "result/BA-n=128-k=3-seed=(1,1)/GRDY/140.mtx",
#     "result/BA-n=128-k=3-seed=(1,1)/L2A-G/595.mtx",
# ]  # first graph is original graph

graph_des = "ER-n=256-p=3.44e-02-seed=(1,1)"
files = [
    "result/ER-n=256-p=3.44e-02-seed=(1,1)/GRDY/0.mtx",
    "result/ER-n=256-p=3.44e-02-seed=(1,1)/GRDY/200.mtx",
    "result/ER-n=256-p=3.44e-02-seed=(1,1)/ModA-G/500.mtx",
    "result/ER-n=256-p=3.44e-02-seed=(1,1)/L2A-G/2007.mtx",
]
G = [read_Graph(file) for file in files]
# print(
#     np.max(np.linalg.eigvals(G_org)),
#     np.max(np.linalg.eigvals(G_ModA)),
#     np.max(np.linalg.eigvals(G_swt)),
# )
# cmap = colors.ListedColormap(['black', 'white'])
# plt.imshow(G_swt) #, cmap=mpl.colormaps['Greys']
# plt.show()

lambda1 = np.max(np.real(np.linalg.eigvals(G[0])))

fig, ax = plt.subplots()
N = G[0].shape[0]

k = len(files)
split = 1.5
betaCnt = 40
tmax = 500
maxepoch = 100
p1 = np.int64(np.floor(betaCnt * 1 / 2))
p2 = betaCnt - p1 + 1
betaList = np.hstack((np.linspace(1e-9, split, p1)[:-1], np.linspace(split, 5, p2)))
measurements = {}
lifespan = np.zeros((k, betaCnt, 2))
coverageMean = np.zeros((k, betaCnt, 2))
coverageVar = np.zeros((k, betaCnt, 2))
graphLabels = np.arange(1, k + 1)
endemicFlag = np.zeros(k)
scale = np.sqrt(10000)
for i in range(betaCnt):
    print("{:3d}: {:.2f}".format(i, betaList[i] / lambda1), end=" ", flush=True)
    for j in range(k):
        measurements[(i, j)] = {
            "graph": graphLabels[j],
            "beta": betaList[i] / lambda1 / scale,
            "mu": 1 / scale,
            "lifespan": [],
            "coverage": [],
            "infect": [],
        }
        for epoch in range(maxepoch):
            SISmodel = SIS(G[j], betaList[i] / lambda1 / scale, 1 / scale, "hub")
            T, I, C, lifespan = SISmodel.Gillespie(tmax, samplingRate=1)
            T, I, C = np.array(T), np.array(I), np.array(C)
            measurements[(i, j)]["coverage"].append(C[-1] / N)
            measurements[(i, j)]["lifespan"].append(lifespan)
            measurements[(i, j)]["infect"].append(np.mean(I) / N)
            if epoch % 50 == 0:
                print(".", end="", flush=True)
        print("{:.2f}".format(np.mean(measurements[(i, j)]["coverage"])), end=", ")
    print("")

cmap = mpl.colormaps["plasma_r"]
color = cmap(np.linspace(0, 1, k))
for i in range(betaCnt):
    for j in range(k):
        if i == 0:
            ax.scatter(
                betaList[i] / lambda1,
                np.mean(measurements[(i, j)]["coverage"]) / N,
                color=color[j],
                label=graphLabels[j],
            )
        else:
            ax.scatter(
                betaList[i] / lambda1,
                np.mean(measurements[(i, j)]["coverage"]) / N,
                color=color[j],
                label="",
            )
    # idx = lifespan[j,:,1] > maxepoch*admitfrac
    # ax.plot(betaList[idx]/lambda1,lifespan[j,idx,0]/lifespan[j,idx,1],label = graphLabels[j])

ax.legend()
fig.savefig("test.pdf")

with open(
    "result/{}/data-{}-{}-{}.pkl".format(graph_des, betaCnt, tmax, maxepoch),
    "wb",
) as f:
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
