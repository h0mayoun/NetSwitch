import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib import colors
from NetSwitchAlgsMod import NetSwitch
import csv
import sys
import matplotlib.animation as animation
import os
import time

random.seed(1)
np.random.seed(1)

n = 16
p = np.log2(n) * 1.1 / n
graph = ig.Graph.Erdos_Renyi(n=n, p=p)

G = NetSwitch(graph)

D = G.deg
m = sum(D)
for i in range(1, n):
    print(D - (D[i - 1] + D[i]) / 2)
