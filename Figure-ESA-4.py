import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter


with open("./Critical_coupling_singleER_N2to15_p50byN_new.pkl", "rb") as in_f:
    data = pickle.load(in_f)

labels = ["ORG", "SWPC", "GRDY"]
colors = ["#1f78b4", "#33a02c", "#e31a1c"]
fig, ax1 = plt.subplots(figsize=(5, 3))
for label, color in zip(labels, colors):
    (couplings, r, stdr, kc) = data[label]
    ax1.plot(
        couplings,
        r,
        color=color,
        ls="-",
        label="Order parameter",
    )

plt.savefig("ESA-Fig-4.pdf", dpi=300)
