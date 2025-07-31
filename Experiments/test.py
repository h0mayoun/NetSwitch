import numpy as np
from NetSwitchAlgs import *
import matplotlib.pyplot as plt
import pickle
import time
from numba import jit



start = time.time()

sample_count = 10000
#np.random.seed(0)
net = ig.Graph.Erdos_Renyi(n=512, p=.05)
Snet = NetSwitch(net, pos_only=False)
print(Snet.get_edges())
end = time.time()
print(end-start)
1/0


avg_checkerCount = []
for r_min in range(40):
    target_r = [r_min*.05-1, r_min*.05-.95]
    print('Target r range:', target_r)
    ''' Load the network and make SwitchNet Obj'''

    # filename = 'configs_custom_n9_m9_noDB.pkl'
    # with open(filename, 'rb') as in_f:
    #     data = pickle.load(in_f)

    Snet = NetSwitch(net, pos_only=False)

    ''' Sampling Procedure'''
    cur_r = Snet.assortativity_coeff()
    while not target_r[0] <= cur_r <= target_r[1]:
        swt = Snet.find_random_checker(pos=True if cur_r < target_r[0] else False)
        Snet.switch(swt)
        cur_r = Snet.assortativity_coeff()
    print('Target range reached, now sampling...')

    x_pos, x_neg = Snet.total_checkers(pos=True), Snet.total_checkers(pos=False)
    cur_s = x_pos + x_neg
    posSwt_ratio = x_pos / cur_s
    samples = [cur_r]
    iterNo = 0
    while iterNo < sample_count:
        swt = Snet.find_random_checker(pos=True if np.random.rand()<posSwt_ratio else False)
        Snet.switch(swt)
        nxt_r = Snet.assortativity_coeff()
        nxt_s = Snet.total_checkers(both=True)

        if not target_r[0] <= nxt_r <= target_r[1]:
            accept_pr = 0
        else:
            accept_pr = min([1, cur_s / nxt_s])

        if np.random.rand() >= accept_pr:
            Snet.switch(swt)
        else:
            cur_r = nxt_r
            cur_s = nxt_s
            posSwt_ratio = Snet.total_checkers(pos=True) / cur_s
            samples.append(cur_s)
            iterNo += 1
            if iterNo % 1000 == 0:
                print(iterNo, ', ', end='')
    print('')
    avg_checkerCount.append((np.mean(target_r), np.mean(samples)))
plt.figure()
min_checker_count = min([i[1] for i in avg_checkerCount])
#plt.hist(samples, bins=100)
plt.plot([i[0] for i in avg_checkerCount], [i[1]/min_checker_count for i in avg_checkerCount])
plt.xlabel('Assortativity')
plt.ylabel('Sampling weight per state')
plt.show()
end = time.time()
print(end-start)