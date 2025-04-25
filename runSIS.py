from SIS import *
from NetSwitchAlgsMod import *

from readGraph import read_Graph

def getSS(T,I,mxTime):
    T = np.array(T)
    I = np.array(I)
    if T[-1] >= mxTime:
        return np.mean(I[T>mxTime*0.9])
    else:
        return 0
    
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

G_org = read_Graph("graphs/" + file[filenum])
G_ModA = read_Graph("result/" + file[filenum] + "/ModA-G/3233.mtx")
G_swt = read_Graph("result/" + file[filenum] + "/GRDY/1565.mtx")

# cmap = colors.ListedColormap(['black', 'white']) 
# plt.imshow(G_swt) #, cmap=mpl.colormaps['Greys']
# plt.show()

lambda1 = np.max(np.real(np.linalg.eigvals(G_org)))

fig, ax = plt.subplots()
#r0List = np.linspace(0.01,0.2,39)
r0List = np.linspace(0.01,0.3,59)
beta,i0 = 1/lambda1, 0.01
N = G_org.shape[0]



mxTime = 10000

for r0 in r0List:
    Ilog = np.zeros((4,0))
    for i in range(100):
        mu = beta/r0
        print(i,end = " ",flush=True)
        infected = int(np.floor(i0 * N))
        infected = np.random.permutation([1] * infected + [0] * (N - infected)).tolist()
        SISmodelSwt = SIS(G_swt, beta, mu, infected)
        SISmodelSwt.step_simulation(mxTime)
        SwtICnt = getSS(SISmodelSwt.Ts,SISmodelSwt.Is,mxTime)
        SISmodelModA = SIS(G_ModA, beta, mu, infected)
        SISmodelModA.step_simulation(mxTime)
        ModAICnt = getSS(SISmodelModA.Ts,SISmodelModA.Is,mxTime)
        SISmodelOrg = SIS(G_org, beta, mu, infected)
        SISmodelOrg.step_simulation(mxTime)
        OrgICnt = getSS(SISmodelOrg.Ts,SISmodelOrg.Is,mxTime)
        
        Ilog = np.hstack((Ilog,np.array([[beta/mu],[SwtICnt],[ModAICnt],[OrgICnt]])))
    mn = np.mean(Ilog,axis = 1)
    print("\n",beta/mu,mn)
    ax.scatter([beta/mu],[mn[1]/N],color = "r")
    ax.scatter([beta/mu],[mn[2]/N],color = "g")
    ax.scatter([beta/mu],[mn[3]/N],color = "b")

fig.savefig(file[filenum] + ".pdf")
# for i in range(100):
#     SISmodel.step_simulation()
#     print(SISmodel.count_infections())
# SISmodel.plot_epi_ts(file[filenum] + ".pdf")
# print(SISmodel.count_infections())
# SISmodel.animate_spread(1, 1000, frame_duration=0.1, gif_name=file[filenum])
# SISmodel.plot_epi_ts(file[filenum] + "org.pdf")
