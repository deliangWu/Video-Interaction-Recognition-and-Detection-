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
            print(seq)
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
    accus = np.mean(accuList_resize,0)
    return (steps,accus)

    

fname = common.path.logPath + 'c3d_finetune_on_ut_set1_dual_nets_04-01-09-41.txt'
steps,accus = readLog(fname)
plt.plot(steps,accus)
    
plt.xlabel('Training steps')
plt.ylabel('Classification accuracy')
ax = plt.gca()
ax.set_xticks(np.arange(0, 3201, 400))
ax.set_yticks(np.arange(0, 1.1, 0.1))
plt.grid(b=1)
plt.show()
