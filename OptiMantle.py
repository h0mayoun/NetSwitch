import igraph as ig
import networkx as nx
import numpy as np
import random
import time
from matplotlib import rc
import matplotlib.pyplot as plt

import matplotlib as mpl

# random.seed(1)
# rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc("text", usetex=True)


class OptiMantle:
    def __init__(self, G):
        self.n = G.vcount()
        self.n_initial = self.n
        self.D = np.array(G.degree(), dtype=np.int64)
        self.A = np.array(G.get_adjacency().data, dtype=np.int8)
        self.sort(-self.D)
        self.iG = ig.Graph.Adjacency(self.A)
        self.Csizes = np.array(self.iG.connected_components().sizes())

    def sort(self, order):
        sortIdx = np.argsort(order)
        self.A = self.A[sortIdx, :][:, sortIdx]
        self.D = self.D[sortIdx]

    def delete_vertices(self, i):
        self.n = self.n - 1
        self.iG.delete_vertices(i)
        self.D = np.array(self.iG.degree(), dtype=np.int64)

    def get_betweeness(self):
        return self.iG.betweenness()

    def get_eigen_centrality(self):
        option = ig.ARPACKOptions()
        option.maxiter = 10000
        try:
            eigc = self.iG.eigenvector_centrality(arpack_options=option)
        except:
            print("Eigensolver failed, resort to degree sequence")
            eigc = self.D
        return eigc

    def get_statistic(self):
        self.Csizes = np.array(self.iG.connected_components().sizes())
        if self.n <= 1:
            return 0, 0
        return max(self.Csizes) / self.n_initial, np.dot(
            self.Csizes, self.Csizes - 1
        ) / (self.n_initial * (self.n_initial - 1))

    def dismantle(self, alg="rand"):
        LCC_series = np.zeros(self.n)
        connectivity_series = np.zeros(self.n)
        timelog = np.zeros(self.n)
        n_initial = self.n

        if alg.lower() == "deg":
            self.sort(-self.D)
        elif alg.lower() == "betw":
            self.sort(-np.array(self.get_betweeness()))
        elif alg.lower() == "eigc":
            self.sort(-np.array(self.get_eigen_centrality()))

        while self.n > 0:
            lcc, con = self.get_statistic()
            LCC_series[n_initial - self.n] = lcc
            connectivity_series[n_initial - self.n] = con

            start_time = time.time()
            if alg.lower() == "rand":
                i = random.randint(0, self.n - 1)
                self.delete_vertices(i)
            elif alg.lower() == "deg":
                self.delete_vertices(0)
            elif alg.lower() == "deg-a":
                i = np.argmax(self.D)
                self.delete_vertices(i)
            elif alg.lower() == "betw":
                self.delete_vertices(0)
            elif alg.lower() == "betw-a":
                i = np.argmax(self.get_betweeness())
                self.delete_vertices(i)
            elif alg.lower() == "eigc":
                self.delete_vertices(0)
            elif alg.lower() == "eigc-a":
                i = np.argmax(self.get_eigen_centrality())
                self.delete_vertices(i)
            else:
                raise Exception("Invalid Algorithm")
            timelog[n_initial - self.n - 1] = time.time() - start_time

        return (LCC_series, connectivity_series, timelog)


# Example usage
if __name__ == "__main__":
    fig, ax = plt.subplots(3, 1, figsize=(6, 6))
    n = 128

    alg = ["deg", "deg-a", "betw", "betw-a", "eigc", "eigc-a", "rand"]
    avg = np.zeros((n, 3, len(alg)))
    color = mpl.cm.get_cmap("tab20")
    label = [
        "Degree",
        "Degree-A",
        "Between",
        "Between-A",
        "EigCen",
        "EigCen-A",
        "Random",
    ]

    k = 120
    for i in range(k):
        print(i)
        for j in range(len(alg)):
            if i < k / 3:
                iGraph = ig.Graph.Erdos_Renyi(n, 6 / n, directed=False)
            elif i < 2 * k / 3:
                iGraph = ig.Graph.Barabasi(n, 3, directed=False)
            else:
                iGraph = ig.Graph.Watts_Strogatz(1, n, 3, 0)

            G = OptiMantle(iGraph)
            lcc, con, timelog = G.dismantle(alg=alg[j])
            ax[0].plot(lcc, color=color(j / 20), label="", alpha=1 / k)
            ax[1].plot(con, color=color(j / 20), label="", alpha=1 / k)
            ax[2].plot(
                np.arange(n, 0, -1), timelog, color=color(j / 20), label="", alpha=1 / k
            )
            avg[:, 0, j] = avg[:, 0, j] + 1 / k * lcc
            avg[:, 1, j] = avg[:, 1, j] + 1 / k * con
            avg[:, 2, j] = avg[:, 2, j] + 1 / k * timelog

    for j in range(len(alg)):
        ax[0].plot(avg[:, 0, j], color=color(j / 20), label=label[j])
        ax[1].plot(avg[:, 1, j], color=color(j / 20), label=label[j])
        ax[2].plot(
            np.arange(n, 0, -1), avg[:, 2, j], color=color(j / 20), label=label[j]
        )

    print(np.sum(avg, axis=0))

    ax[0].set_xlim([0, n])
    ax[1].set_xlim([0, n])
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[2].spines["top"].set_visible(False)
    ax[2].spines["right"].set_visible(False)
    ax[2].set_yscale("log")
    ax[2].set_xscale("log")
    ax[2].set_xscale("log")
    # ax[0].set_xlabel("Vertices removed")
    ax[2].legend(loc="upper center", frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.3))
    ax[0].set_ylabel("LCC size")
    ax[1].set_ylabel("Connecticity")
    ax[2].set_ylabel("Time")
    ax[2].set_xlabel("Vertices removed")
    plt.tight_layout()
    plt.savefig("dismantle-" + str(n) + ".pdf")
