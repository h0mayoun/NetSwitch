import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import colorsys

# Input HSL:
h = 200 / 360  # Hue: [0,1]
l = 0.50  # Lightness: [0,1]
s = 0.75  # Saturation: [0,1]

# colorsys uses H, L, S ordering:
r, g, b = colorsys.hls_to_rgb(h, l, s)
# Define a list of colors from light to dark


n = 7
offset = 1 / n * 0.1
colors = [colorsys.hls_to_rgb(1 / n * i + offset, 0.7, 0.9) for i in range(n)]
print(colors)
# colors = ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]

# Create the colormap
cmap = mcolors.LinearSegmentedColormap.from_list("OpTiX", colors, N=n)

# Plot
x = np.linspace(-1, 1, 50 * n)
y = np.linspace(-1, 1, 50 * n)
X, Y = np.meshgrid(x, y, indexing="xy")
data = X**2 - Y**2
plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.savefig("OpTiX.pdf")
