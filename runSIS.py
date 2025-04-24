from SIS import *
from NetSwitchAlgsMod import *

from readGraph import read_Graph

file = [
    "email-enron-only.mtx",
    "reptilia-tortoise-network-bsv.edges",
    "inf-USAir97.mtx",
    "aves-wildbird-network.edges",
    "ca-netscience.mtx",
    "ia-radoslaw-email.edges",
]

filenum = 0

G_org = read_Graph("graphs/" + file[filenum])
G_swtA = read_Graph("result/" + file[filenum] + "/ModA-G/600.mtx")
G_swt = read_Graph("result/" + file[filenum] + "/GRDY/583.mtx")

# cmap = colors.ListedColormap(['black', 'white'])
# plt.imshow(G_swt) #, cmap=mpl.colormaps['Greys']
# plt.show()

lambda1 = np.max(np.real(np.linalg.eigvals(G_org)))

fig, ax = plt.subplots()
beta, mu, i0 = 0.9 / lambda1, 0.6, 0.05

SISmodelSwt = SIS(G_swt, beta, mu, i0)
SISmodelSwt.step_simulation(1000)
SISmodelSwt.plot_epi_ts(ax=ax, label="GRDY")
SISmodelSwtA = SIS(G_swtA, beta, mu, i0)
SISmodelSwtA.step_simulation(1000)
SISmodelSwtA.plot_epi_ts(ax=ax, label="ModA-G")
SISmodelOrg = SIS(G_org, beta, mu, i0)
SISmodelOrg.step_simulation(1000)
SISmodelOrg.plot_epi_ts(ax=ax, label="ORG")
fig.savefig(file[filenum] + ".pdf")
# for i in range(100):
#     SISmodel.step_simulation()
#     print(SISmodel.count_infections())
# SISmodel.plot_epi_ts(file[filenum] + ".pdf")
# print(SISmodel.count_infections())
# SISmodel.animate_spread(1, 1000, frame_duration=0.1, gif_name=file[filenum])
# SISmodel.plot_epi_ts(file[filenum] + "org.pdf")
