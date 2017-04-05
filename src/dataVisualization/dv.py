from __future__ import print_function
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import sys
sys.path.insert(1,'../common')
import common

def readLog(logName,seqBias = 1):
    with open(logName) as f:
        content = f.readlines()
    stepList = []
    accuList = []
    seq = 0
    for line in content:
        line = line.strip()
        if 'current sequence' in line:
            seq = int(line[line.index('is') + 3:]) - seqBias 
            stepList.append([])
            accuList.append([])
        elif 'step' in line:
            step = int(line[line.index('step') + 5:line.index(',')])
            stepList[seq].append(step)
            accu = float(line[line.index('anv:') + 5: line.index('anv:') + line[line.index('anv:'):].index(',')])
            accuList[seq].append(accu)
    
    lenList = [len(steps) for steps in stepList]
    maxLen = max(lenList)
    steps = stepList[lenList.index(maxLen)]
    accuList_resize = list(accuList)
    for accus in accuList_resize:
        accus.extend([accus[-1]]*(maxLen-len(accus)))
    accuList_resize = np.array(accuList_resize)
    assert accuList_resize.shape[0] == 10
    accus = np.mean(accuList_resize,0)
    print(accus[-1])
    return (steps,accus)

    

fname_set1_g      = common.path.logPath + 'c3d_train_on_ut_set1_04-01-18-57.txt'
fname_set1_a_s    = common.path.logPath + 'c3d_finetune_on_ut_single_net04-01-14-01.txt'
fname_set1_a_d_s  = common.path.logPath + 'c3d_finetune_on_ut_set1_dual_nets_shareVars_04-01-13-22.txt'
fname_set1_a_d_us = common.path.logPath + 'c3d_finetune_on_ut_set1_dual_nets_04-01-09-41.txt'
steps_set1_g, accus_set1_g      = readLog(fname_set1_g)
steps_set1_as, accus_set1_as    = readLog(fname_set1_a_s)
steps_set1_ads, accus_set1_ads  = readLog(fname_set1_a_d_s)
steps_set1_adus,accus_set1_adus = readLog(fname_set1_a_d_us)


fname_set2_g   = common.path.logPath + 'c3d_train_on_ut_set2_04-01-21-44.txt'
fname_set2_a_s = common.path.logPath + 'c3d_finetune_on_ut_single_net_set2_04-02-05-38.txt'
fname_set2_a_d_s = common.path.logPath + 'c3d_finetune_on_ut_dual_nets_shareVars_set2_04-02-10-25.txt'
fname_set2_a_d_us = common.path.logPath + 'c3d_finetune_on_ut_dual_nets_unShareVars_set2_04-02-10-44.txt'
steps_set2_g,  accus_set2_g  = readLog(fname_set2_g,seqBias=11)
steps_set2_as, accus_set2_as = readLog(fname_set2_a_s,seqBias=11)
steps_set2_ads,  accus_set2_ads  = readLog(fname_set2_a_d_s,seqBias=11)
steps_set2_adus,  accus_set2_adus  = readLog(fname_set2_a_d_us,seqBias=11)

#plt.plot(steps_set1_g,   accus_set1_g,'r--',    label = 'set1 model #1')
#plt.plot(steps_set1_as,  accus_set1_as,'g--',   label = 'set1 model #2') 
#plt.plot(steps_set1_ads, accus_set1_ads,'b--',  label = 'set1 model #3') 
#plt.plot(steps_set1_adus,accus_set1_adus,'c--', label = 'set1 model #4')

plt.plot(steps_set2_g,   accus_set2_g,'r-',    label = 'set2 model #1')
plt.plot(steps_set2_as,  accus_set2_as,'g-',   label = 'set2 model #2') 
plt.plot(steps_set2_ads, accus_set2_ads,'b-',  label = 'set2 model #3') 
plt.plot(steps_set2_adus,accus_set2_adus,'c-', label = 'set2 model #4')
    
plt.xlabel('Training steps')
plt.ylabel('Classification accuracy')
ax = plt.gca()
ax.set_xticks(np.arange(0, 3601, 400))
ax.set_yticks(np.arange(0, 1.1, 0.05))
plt.legend()
plt.grid(b=1)
plt.show()
