import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import igraph as ig
from stdog.utils.misc import (
    ig2sparse,
)  # Function to convert igraph format to sparse matrix
from stdog.dynamics.kuramoto import Heuns
from scipy.sparse.linalg import eigsh
import pickle
from readGraph import *
import sys

graphDef = sys.argv[1]
id1, id2, id3 = sys.argv[2], sys.argv[3], sys.argv[4]
print(sys.argv)
Gs = [
    read_Graph("result/{}/SWPC/{}.mtx".format(graphDef, id1)),
    read_Graph("result/{}/SWPC/{}.mtx".format(graphDef, id2)),
    read_Graph("result/{}/GRDY/{}.mtx".format(graphDef, id3)),
]
N = Gs[0].shape[0]
labels = ["ORG", "SWPC", "GRDY"]
data = {"ORG": [], "SWPC": [], "GRDY": []}
for graph, label in zip(Gs, labels):

    G = ig.Graph.Adjacency(graph)
    adj = ig2sparse(G)
    w, _ = eigsh(adj, k=1, which="LA")
    Kc_theoretical = (2 * np.sqrt(2 * np.pi)) / (np.pi * w[0])
    # print("ER N,p,Kc:", N, ERp, Kc_theoretical)

    # Parameter setting to run Kuramoto simulations
    omegas = np.random.normal(size=N).astype("float32")
    num_mid_couplings = 50
    num_couplings = 80
    coup_min, coup_mid, coup_max = 0.001, 0.12, 0.3
    couplings = np.hstack(
        (
            np.linspace(coup_min, coup_mid, num_mid_couplings + 1)[:-1],
            np.linspace(coup_mid, coup_max, num_couplings - num_mid_couplings),
        )
    )
    phases = np.array(
        [np.random.uniform(-np.pi, np.pi, N) for i_l in range(num_couplings)],
        dtype=np.float32,
    )
    precision = 32
    dt = 0.01
    num_temps = 1000 / dt
    total_time = dt * num_temps
    total_time_transient = total_time
    transient = False

    # Building and running Kuramoto model
    heuns_0 = Heuns(
        adj,
        phases,
        omegas,
        couplings,
        total_time,
        dt,
        device="/gpu:0",  # or /cpu:
        precision=precision,
        transient=transient,
    )

    heuns_0.run()
    heuns_0.transient = True
    heuns_0.total_time = total_time_transient
    heuns_0.run()
    order_parameter_list = (
        heuns_0.order_parameter_list
    )  # (num_couplings, total_time//dt)

    r = np.mean(order_parameter_list, axis=1)
    stdr = np.std(order_parameter_list, axis=1)

    # print(r)
    # print(couplings)

    # print("Numerical K_c:", couplings[np.argmax(stdr)])
    data[label] = [couplings, r, stdr, Kc_theoretical]
# couplings = np.linspace(.02, 0.04, 20)
# r = np.random.random(20)
# stdr = np.random.random(20)
# Kc_theoretical = 0.3
print(data)
with open("Critical_coupling_singleER_N2to15_p50byN_new.pkl", "wb") as out_f:
    pickle.dump(data, out_f)
