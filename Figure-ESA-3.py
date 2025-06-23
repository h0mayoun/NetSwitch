import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to render text
        "font.family": "serif",  # Use serif fonts (like LaTeX default)
        "font.serif": ["Computer Modern Roman"],  # Specify LaTeX font family
    }
)
np.set_printoptions(precision=1, suppress=True)

graphDef = sys.argv[1]
rho = sys.argv[2]

with open("result/{}/{}.pkl".format(graphDef, rho), "rb") as f:
    data = pickle.load(f)

stepCnt = 4
steps = np.linspace(0.35, 0.95, stepCnt)
fig = plt.figure(figsize=(16, 4))
ax = [
    fig.add_subplot(1, 4, 1),
    fig.add_subplot(1, 4, 2),
    fig.add_subplot(1, 4, 3),
    fig.add_subplot(1, 4, 4),
]

color = {"ORG": "#1f78b4", "GRDY": "#e31a1c", "SWPC": "#33a02c"}
labels = {"ORG": "ORG", "GRDY": "SW-NA", "SWPC": "SW-A"}
betaSweep = 1
plotData = [
    {"ORG": [[], []], "GRDY": [[], []], "SWPC": [[], []]},
    {"ORG": [[], []], "GRDY": [[], []], "SWPC": [[], []]},
]
for graph, graphData in data.items():
    # print(graph)
    for beta, betaData in graphData.items():
        # print("\t", beta)
        runs = len(betaData["Ts"])
        # if beta < betaSweep or beta > betaSweep + 0.1:
        #    continue
        notEndemic = np.zeros(runs, dtype=bool)
        CsEndemic = []
        for i in range(runs):
            if betaData["Is"][i][-1] == 0:
                notEndemic[i] = True
                continue
            Ci = betaData["Cs"][i]
            if len(CsEndemic) > 0 and len(Ci) < len(CsEndemic[-1]):
                Ci = np.pad(
                    Ci,
                    (0, len(CsEndemic[-1]) - len(Ci)),
                    mode="edge",
                )
            elif len(CsEndemic) > 0 and len(CsEndemic[-1]) < len(Ci):
                for j, Csj in enumerate(CsEndemic):
                    CsEndemic[j] = np.pad(Csj, (0, len(Ci) - len(Csj)), mode="edge")
            CsEndemic.append(Ci)

        plotData[0][graph][0].append(beta / 36.078080919949684)
        plotData[0][graph][1].append(
            np.mean([x[-1] for x, m in zip(betaData["Ts"], notEndemic) if m])
        )
        if len(CsEndemic) > 0:
            Cmean = np.mean(CsEndemic, axis=0)
            idx = np.searchsorted(Cmean, steps, side="left")
            # print(idx)
            plotData[1][graph][0].append(beta / 36.078080919949684)
            plotData[1][graph][1].append(idx)
        # ax[1].plot(betaData["Ts"][i], betaData["Cs"][i], c=color[graph], lw=1)
# print(plotData[0])
plotted = False
for graph, graphData in plotData[0].items():
    # ax[0].plot(
    #     plotData[0][graph][0],
    #     plotData[0][graph][1],
    #     c=color[graph],
    #     label=labels[graph],
    # )
    #if graph == "SWPC":
    #   continue
    y = np.array(plotData[1][graph][1])
    print(y)
    for i in range(stepCnt):
        ax[i].plot(
            plotData[1][graph][0],
            [x[i] for x in y],
            c=color[graph],
            label=labels[graph]
        )
    plotted = True

ax[0].legend(frameon=False)


ax[0].set_ylabel("Time")

#ax[1].set_title(r"$\rho$-Coverage Time")
ax[1].set_ylabel("Time")
for i in range(stepCnt):
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
    ax[i].set_xlabel(r"$R_0$")
    ax[i].set_title(r"{:.2f}-Coverage Time".format(steps[i]))
    
plt.tight_layout()
plt.savefig("ESA-Fig-3-b.png", dpi=300)
