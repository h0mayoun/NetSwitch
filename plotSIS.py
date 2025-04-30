import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
graphLabels = ["org", "mod", "swt"]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
color = "rgb"
with open(
    "result/{}/data-{}-{}-{}.pkl".format(
        file[int(sys.argv[1])], sys.argv[2], sys.argv[3], sys.argv[4]
    ),
    "rb",
) as f:
    data = pickle.load(f)

plotData = {
    "lambda": [],
    "org": {"lifespan": [], "coverage": []},
    "swt": {"lifespan": [], "coverage": []},
    "mod": {"lifespan": [], "coverage": []},
}

for key, value in data.items():
    # print(key, value)
    # print(value["beta"])
    # print(value["mu"])
    # print(value["lifespan"])
    # print(value["coverage"])
    # print(value["infect"])

    graph = value["graph"]
    lifespan = np.array(value["lifespan"])
    coverage = np.array(value["coverage"])
    finalval = np.array(value["infect"])
    beta = value["beta"]
    mu = value["mu"]

    # meanLifespan = np.mean(lifespan[finalval == 0])
    # print(np.sum(np.isfinite(lifespan)))
    meanLifespan = np.mean(lifespan[np.isfinite(lifespan)])
    meanCoverage = np.mean(coverage)
    # print(beta / mu)
    if len(plotData["lambda"]) == 0 or beta / mu != plotData["lambda"][-1]:
        plotData["lambda"].append(beta / mu)
    plotData[graph]["lifespan"].append(meanLifespan)
    plotData[graph]["coverage"].append(meanCoverage)
    # print(beta / mu, meanLifespan, key[1])
    # ax.scatter(
    #     beta / mu, meanLifespan, color=color[key[1]], label=graphLabels[key[1]]
    # )
# print(plotData)
for i, graph in enumerate(graphLabels):

    y = np.array(plotData[graph]["lifespan"])
    x = np.array(plotData["lambda"])
    idx = np.isfinite(y)
    # print(x, y)
    ax1.plot(x[idx], y[idx], label="", color=color[i], alpha=0.1)
    ax1.plot(x[idx], gaussian_filter1d(y[idx], sigma=3), label=graph, color=color[i])

    y = np.array(plotData[graph]["coverage"])
    x = np.array(plotData["lambda"])
    idx = np.isfinite(y)

    ax2.plot(x[idx], y[idx], label=graph, color=color[i], ls="--")
plt.legend()
plt.savefig("result/{}/plot.pdf".format(file[int(sys.argv[1])]))
