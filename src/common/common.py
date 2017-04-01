from __future__ import print_function
import sys
import os
from os.path import isfile, join
import numpy as np
from datetime import datetime

''' a method for print input string on terminal and write it to the file at the same time'''
def pAndWf(fileName, string):
    f = open(join(path.logPath,fileName),'a+')
    print(string,end='')
    f.write(string)
    f.close()
    return None

def clearFile(fileName):
    f = open(join(path.logPath,fileName),'w+')
    f.close()
    return None

def tupleInsert(tIn,ind,obj):
    l = list(tIn)
    l.insert(ind,obj)
    return tuple(l)

class path:
    if os.name == 'nt':
        projectPath = 'D:/Course/Final_Thesis_Project/project/Video-Interaction-Recognition-and-Detection-/'
        ucfPath = "D:/Course/Final_Thesis_Project/project/datasets/UCF101/"
    else:
        projectPath = '/home/wdl/Video-Interaction-Recognition-and-Detection-/'        
        ucfPath = "/home/wdl/3DCNN/datasets/UCF101/"        

    utSet1Path = join(projectPath, 'datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/')
    utSet1_a0_Path = join(utSet1Path, 'vOut_0/')
    utSet1_a1_Path = join(utSet1Path, 'vOut_1/')

    utSet2Path = join(projectPath, 'datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2/')
    utSet2_a0_Path = join(utSet2Path, 'vOut_0/')
    utSet2_a1_Path = join(utSet2Path, 'vOut_1/')
        
        
    variablePath = join(projectPath,'variableSave/')
    logPath = join(projectPath,'log/')
    
class Vars:
    if os.name == 'nt':
        dev = ['/cpu:0']
    else:
        dev = ['/gpu:1','/gpu:1']

def getDateTime():
    t = str(datetime.now())
    t = t[5:16]
    t = t.replace(' ','-')
    t = t.replace(':','-')
    return t