import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import colors
from local.NetSwitchAlgsMod import *
import copy
import matplotlib.patches as patches

seed1 = 1
seed2 = 1
plt.rcParams.update(
    {"text.usetex": True, "font.family": "STIXGeneral", "mathtext.fontset": "stix"}
)
np.set_printoptions(precision=2, suppress=True, linewidth=2048)
# np.random.seed(seed1)
# random.seed(seed2)
n = 8
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
g = ig.Graph.Erdos_Renyi(n=n, m=10, directed=False)

G = NetSwitch(g,sortAdj=False)

cmap = colors.ListedColormap(["white", "black", "#33a02c", "#e31a1c"])
i, j, k, l = G.find_random_checker()
print(i, j, k, l)


red_id = [(i, l), (j, k)]
green_id = [(i, k), (j, l)]

for u in range(n):
    for v in range(n):
        color = "white"
        if G.A[u, v]:
            color = "black"
            if (u, v) in red_id:# or (v, u) in red_id:
                color = "#e31a1c"
            elif (u, v) in green_id:# or (v, u) in green_id:
                color = "#33a02c"

        rect = patches.Rectangle((v, n - 1 - u), 1, 1, facecolor=color)
        ax[0].add_patch(rect)

ax[0].set_xlim(0, n)
ax[0].set_ylim(0, n)
ax[0].set_xticks([])
ax[0].set_yticks([])

G.switch((i, j, k, l))

for u in range(n):
    for v in range(n):
        color = "white"
        if G.A[u, v]:
            color = "black"
            if (u, v) in red_id:# or (v, u) in red_id:
                color = "#e31a1c"
            elif (u, v) in green_id:# or (v, u) in green_id:
                color = "#33a02c"

        rect = patches.Rectangle((v, n - 1 - u), 1, 1, facecolor=color)
        ax[2].add_patch(rect)

ax[2].set_xlim(0, n)
ax[2].set_ylim(0, n)
ax[2].set_xticks([])
ax[2].set_yticks([])

g = ig.Graph.Adjacency(G.A, mode=ig.ADJ_UNDIRECTED)
layout = g.layout("fr")
visual_style = {
    "vertex_size": 60,  # smaller vertex size
    "vertex_color": "white",  # white fill
    "vertex_label_color": "black",  # label color for contrast
    "layout": layout,  # Fruchterman-Reingold layout
    "bbox": (300, 300),  # bounding box size
    "margin": 20,
}
print(i, j, k, l)
g.add_edge(i, l)
g.add_edge(j, k)
red_id = [g.get_eid(i, l), g.get_eid(j, k)]
green_id = [g.get_eid(i, k), g.get_eid(j, l)]
g.es["color"] = [
    "#e31a1c" if e.index in red_id else "#33a02c" if e.index in green_id else "black"
    for e in g.es
]
g.vs["label"] = g.vs.indices
ig.plot(g, target=ax[1], **visual_style)

rect = patches.Rectangle(
    (k+0.5, n-j-0.5),         # (x, y)
    l-k, j-i,           # width, height
    linewidth=2,
    edgecolor='grey',
    facecolor='none',  # transparent fill
    ls ="dotted"
)

ax[0].add_patch(rect)
rect = patches.Rectangle(
    (k+0.5, n-j-0.5),         # (x, y)
    l-k, j-i,           # width, height
    linewidth=2,
    edgecolor='grey',
    facecolor='none',  # transparent fill
    ls ="dotted"
)
ax[2].add_patch(rect)
ax[2].set_title("Positive Chekerboard")
ax[0].set_title("Negative Chekerboard")
plt.tight_layout()
plt.savefig("ESA-Fig-6.png", dpi=300)
