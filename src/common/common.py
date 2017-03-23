from __future__ import print_function
import sys
import os
from os.path import isfile, join
import numpy as np

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

class path:
    if os.name == 'nt':
        projectPath = 'D:/Course/Final_Thesis_Project/project/Video-Interaction-Recognition-and-Detection-/'
        ucfPath = "D:/Course/Final_Thesis_Project/project/datasets/UCF101/"
        utSet1Path = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/"
        utSet2Path = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2/"
    else:
        projectPath = '/home/wdl/Video-Interaction-Recognition-and-Detection-/'        
        ucfPath = "/home/wdl/3DCNN/datasets/UCF101/"        
        utSet1Path = "/home/wdl/3DCNN/datasets/ut_interaction/segmented_set1/"
        utSet2Path = "/home/wdl/3DCNN/datasets/ut_interaction/segmented_set2/"
        
    variablePath = join(projectPath,'variableSave/')
    logPath = join(projectPath,'log/')