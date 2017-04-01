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

fname = common.path.logPath + 'c3d_finetune_on_ut_set1_dual_nets_04-01-09-41.txt'
with open(fname) as f:
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
        
plt.plot(stepList[0],accuList[0],stepList[1],accuList[1],stepList[2],accuList[2], stepList[3],accuList[3], stepList[4],accuList[4])
plt.xlim((0,2000))
plt.ylim((0,1))
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.show()
