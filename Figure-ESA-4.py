import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


with open("./Critical_coupling_singleER_N2to15_p50byN_new.pkl", "rb") as in_f:
    data = pickle.load(in_f)

graphs = ["ORG", "GRDY", "SWPC"]
labels = ["ORG", "SW-NA", "SW-A"]
colors = ["#1f78b4", "#e31a1c", "#33a02c"]
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax02 = ax[0].twinx()
for graph, label, color in zip(graphs, labels, colors):
    if graph == "SWPC":
        continue
    (couplings, r, stdr, kc) = data[graph]
    ax[1].plot(
        couplings,
        r,
        color=color,
        ls="-",
        label="",
    )
    ax[0].plot(
        couplings,
        savgol_filter(stdr**2 / r, 8, 1),
        color=color,
        ls="-",
        label=label,
    )
    # ax[0].plot(
    #     couplings,
    #     r,
    #     color=color,
    #     ls="-",
    #     label="",
    # )

ax[0].set_xlim(0.0, 0.08)
ax[1].set_xlim(0.0, 0.2)

ax[0].legend(frameon=False)
ax[0].set_ylabel(r"$\tilde{\chi}(R)$")
# ax02.set_ylabel(r"$\chi(R)$")
ax[1].set_ylabel(r"$\langle R \rangle_t$")

ax[0].set_xlabel(r"$K$")
ax[1].set_xlabel(r"$K$")
mn, mx = ax[0].get_ylim()
# for label, color in zip(labels, colors):
#     (couplings, r, stdr, kc) = data[label]
#     ax[0].plot([kc, kc], [mn, mx], ls=":", c="k")

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("ESA-Fig-4-b.png", dpi=300)
