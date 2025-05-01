import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import random
import time
import EoN
import networkx as nx
import time
from scipy.sparse.csgraph import laplacian
import imageio
import os
import copy


class SIS:

    def __init__(self, A, beta, mu, i0, population_division=None):
        # alpha = Beta/Mu
        self.A = A
        self.N = A.shape[0]
        self.beta, self.mu = beta, mu
        self.t = 0
        if (type(i0) is float) and (0 < i0 < 1):
            I0 = int(np.floor(i0 * self.N))
            self.I = np.random.permutation([1] * I0 + [0] * (self.N - I0))
        elif type(i0) is list:
            self.I = np.array(i0)
        elif i0 == "hub":
            self.I = np.zeros(self.N)
            self.I[np.argmax(np.sum(self.A, axis=1))] = 1

        # initialize the time/infecteds timeseries
        if population_division is None:
            self.pop_division = None
        else:
            self.pop_division = np.concatenate(
                (0, population_division, self.N), axis=None
            )
        self.Ts = [self.t]
        self.C = copy.copy(self.I)
        self.Is = [self.count_infections()]
        self.Cs = [np.sum(self.C)]

    def count_infections(self):
        if self.pop_division is None:
            return np.sum(self.I)
        else:
            divs = zip(self.pop_division[0:], self.pop_division[1:])
            return tuple([np.sum(self.I[i:j]) for i, j in divs])

    def step_simulation(self, step_size=1, animate=False):
        self.Gillespie(self.t + step_size, animate=animate)

    def plot_epi_ts(self, file=None, ax=None, label="total"):
        plt.figure()
        if self.pop_division is None:
            if ax == None:
                plt.plot(self.Ts, self.Is, label=label)
                plt.legend(frameon=False)
            else:
                ax.plot(self.Ts, self.Is, label=label)
                ax.legend(frameon=False)
        else:
            plt.plot(self.Ts, [np.sum(i) for i in self.Is], label=label)
            for div in range(np.size(self.pop_division) - 1):
                plt.plot(
                    self.Ts,
                    [i[div] for i in self.Is],
                    label=str(self.pop_division[div]),
                )
        # plt.ylim([0, 500])
        if file == None and ax == None:
            plt.show()
        elif ax == None:
            plt.savefig(file, dpi=300)
        return True

    def save_adj_snapshot(self, frameno=0, IS_channels=None, name_prefix=""):
        plt.figure()
        plt.imshow(self.A, cmap="Greys")
        if IS_channels:
            # print(self.active_IS_channels)
            plt.scatter(
                [i[0] for i in self.active_IS_channels],
                [i[1] for i in self.active_IS_channels],
                c="red",
                s=0.5,
                alpha=1,
            )
            plt.scatter(
                [i[1] for i in self.active_IS_channels],
                [i[0] for i in self.active_IS_channels],
                c="red",
                s=0.5,
                alpha=1,
            )
        plt.axis("square")
        plt.xlim([0, self.N - 1])
        plt.ylim([self.N - 1, 0])
        plt.tight_layout()
        plt.savefig(name_prefix + str(int(frameno)) + ".png", dpi=200)
        plt.close()
        return True

    def animate_spread(
        self, step_size=1, max_step=100, frame_duration=0.1, gif_name="SIS"
    ):
        frameno = 0
        self.save_adj_snapshot(frameno, name_prefix=gif_name)
        while int(max_step) > 0:
            self.step_simulation(step_size, animate=True)
            frameno += 1
            self.save_adj_snapshot(frameno, IS_channels=True, name_prefix=gif_name)
            max_step -= step_size
        # Build GIF
        duration = frame_duration * np.ones(frameno + 1)
        with imageio.get_writer(
            gif_name + ".gif", mode="I", duration=list(duration)
        ) as writer:
            for fileNo in range(frameno + 1):
                image = imageio.imread(gif_name + str(fileNo) + ".png")
                writer.append_data(image)
        # deleting frames
        for fileNo in range(frameno + 1):
            os.remove(gif_name + str(fileNo) + ".png")
        return True

    def Gillespie(self, tmax, animate=False):
        if animate:
            self.active_IS_channels = []
        while (self.t <= tmax) and (np.sum(self.Is[-1]) > 0):
            #print("{:.1f}-{:.1f}".format(self.t,),end = " ",flush = True)
            I_idxs = np.where(self.I == 1)[0]
            S_idxs = np.where(self.I == 0)[0]
            IS_mat = self.A[I_idxs, :][:, S_idxs]
            IS_mat_colsum = np.sum(IS_mat, axis=0)
            IS_mat_sum = np.sum(IS_mat_colsum)
            # print(IS_mat_colsum.shape)
            recovery_rate, infection_rate = (
                np.sum(self.I) * self.mu,
                IS_mat_sum * self.beta,
            )
            event_rate = recovery_rate + infection_rate
            delay = -np.log(1.0 - np.random.rand()) / event_rate
            if np.random.rand() < recovery_rate / event_rate:  # One node is recovering
                self.I[random.choice(I_idxs)] = 0
                self.Is.append(self.Is[-1] - 1)
                # self.Is.append(self.count_infections())  # self.Is[-1] - 1
                self.Cs.append(self.Cs[-1])
            else:
                rand_IS_channel = np.random.randint(IS_mat_sum)
                new_infected_node = S_idxs[
                    np.where(np.cumsum(IS_mat_colsum) > rand_IS_channel)[0][0]
                ]
                if animate:
                    spreader_margin = rand_IS_channel - np.sum(
                        IS_mat_colsum[: np.where(S_idxs == new_infected_node)[0][0]]
                    )
                    new_spreader = I_idxs[
                        np.where(
                            IS_mat[:, np.where(S_idxs == new_infected_node)[0][0]] == 1
                        )[0][spreader_margin]
                    ]
                    self.active_IS_channels.append((new_spreader, new_infected_node))
                self.I[new_infected_node] = 1
                self.Is.append(self.Is[-1] + 1)
                # self.Is.append(self.count_infections())  # self.Is[-1] + 1
                Ccur = self.Cs[-1]
                if self.C[new_infected_node] == 0:
                    self.C[new_infected_node] = 1
                    Ccur += 1
                self.Cs.append(Ccur)
            self.t += delay
            self.Ts.append(self.t)

        return self.Ts, self.Is, self.Cs


def poorclub_topnode(g):
    lv, lw = np.linalg.eig(laplacian(g))
    fiedler_vec = lw[:, np.argsort(lv.real)[1]]
    FV_sign = np.divide(fiedler_vec, np.abs(fiedler_vec))
    cluster_nodes = np.where(FV_sign == 1)[0]
    if cluster_nodes[0] == 0:
        cluster_nodes = np.where(FV_sign == -1)[0]
    return np.min(cluster_nodes)


# # with open('../Sims/switch_process/N1000_BA25/SR_N1000_ba25_adj_original.csv', newline='\n') as f_in:
# with open("Switched Nets/N2048_ba50_adj_original.csv", newline="\n") as f_in:
#     reader = csv.reader(f_in)
#     G_org = np.array([[int(i) for i in row] for row in reader])

# with open("Switched Nets/SG_N2048_ba50_adj_switched.csv", newline="\n") as f_in:
#     reader = csv.reader(f_in)
#     G_swt = np.array([[int(i) for i in row] for row in reader])

# G = G_org
# print("hub index", np.argmax(np.sum(G, axis=1)))
# Gnx = nx.from_numpy_array(G)
# # cmap = colors.ListedColormap(['black', 'white'])
# # plt.imshow(G_swt) #, cmap=mpl.colormaps['Greys']
# # plt.show()

# lambda1 = np.max(np.real(np.linalg.eigvals(G_org)))
# print("\lambda_1", lambda1, np.max(np.real(np.linalg.eigvals(G_swt))))

# SISmodel = SIS(G_swt, 1.1 / lambda1, 1, 0.05, poorclub_topnode(G_swt))
# SISmodel.animate_spread(1, 200, frame_duration=0.1, gif_name="BA_swt_net")
# # T, I = SISmodel.Gillespie(100)
# # print(I)
# # SISmodel.plot_epi_ts()
# 1 / 0
# for aGlide in np.linspace(0.95, 1.1, 16):
#     alpha = aGlide / lambda1
#     iterNo = 50
#     endI = []
#     timer = 0
#     for iter in range(iterNo):
#         st = time.time()
#         SISmodel = SIS(G, alpha, 1, 0.1)
#         T, I = SISmodel.Gillespie(100)
#         # if iter == 50 and aGlide>1:
#         #     SISmodel.plot_epi_ts()
#         # t, S, I = EoN.Gillespie_SIS(Gnx, alpha, 1, tmax=100, rho=0.05)
#         timer += time.time() - st
#         endI.append(I[-1])
#     timer /= iterNo
#     print(aGlide, np.mean(endI), round(timer, 2))
