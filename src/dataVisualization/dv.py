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

def readLog(logName):
    with open(logName) as f:
        content = f.readlines()
    stepList = []
    accuList = []
    seq = 0
    for line in content:
        line = line.strip()
        if 'current sequence' in line:
            seq = int(line[line.index('is') + 3:]) - 1
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
    return (steps,accus)

    

fname_a_d_us = common.path.logPath + 'c3d_finetune_on_ut_set1_dual_nets_04-01-09-41.txt'
fname_a_d_s  = common.path.logPath + 'c3d_finetune_on_ut_set1_dual_nets_shareVars_04-01-13-22.txt'
fname_a_s    = common.path.logPath + 'c3d_finetune_on_ut_single_net04-01-14-01.txt'
fname_g      = common.path.logPath + 'c3d_train_on_ut_set1_04-01-18-57.txt'

steps_adus,accus_adus = readLog(fname_a_d_us)
steps_ads, accus_ads  = readLog(fname_a_d_s)
steps_as, accus_as    = readLog(fname_a_s)
steps_g, accus_g      = readLog(fname_g)

plt.plot(steps_ads,accus_ads,'r--', steps_adus,accus_adus,'b-', steps_as, accus_as,'g+', steps_g, accus_g,'r.')
    
plt.xlabel('Training steps')
plt.ylabel('Classification accuracy')
ax = plt.gca()
ax.set_xticks(np.arange(0, 3601, 400))
ax.set_yticks(np.arange(0, 1.1, 0.1))
plt.grid(b=1)
plt.show()
