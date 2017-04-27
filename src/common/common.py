from __future__ import print_function
import sys
import os
from os.path import isfile, join
import numpy as np
from numpy import genfromtxt
from datetime import datetime
import csv

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

def saveList2File(fileName,theList):
    with open(fileName,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(theList)
    f.close()
    return None

def readListFromFile(fileName):
    theList = []
    theList = genfromtxt(fileName,delimiter=',')
    return theList
    
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
    utSet1DetPath = join(projectPath, 'datasets/UT_Interaction/ut-interaction_set1/')

    utSet2Path = join(projectPath, 'datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2/')
    utSet2_a0_Path = join(utSet2Path, 'vOut_0/')
    utSet2_a1_Path = join(utSet2Path, 'vOut_1/')
    utSet2DetPath = join(projectPath, 'datasets/UT_Interaction/ut-interaction_set2/')
        
        
    variablePath = join(projectPath,'variableSave/')
    logPath = join(projectPath,'log/')
    
class Vars:
    if os.name == 'nt':
        dev = ['/cpu:0']
    else:
        dev = ['/gpu:0','/gpu:1']
        
    feature_g_VarsList = ['top/feature_descriptor_g/conv1/weight:0',  'top/feature_descriptor_g/conv1/bias:0', \
                          'top/feature_descriptor_g/conv2/weight:0',  'top/feature_descriptor_g/conv2/bias:0', \
                          'top/feature_descriptor_g/conv3a/weight:0',  'top/feature_descriptor_g/conv3a/bias:0', \
                          'top/feature_descriptor_g/conv4a/weight:0', 'top/feature_descriptor_g/conv4a/bias:0', \
                          #'top/feature_descriptor_g/conv4b/weight:0', 'top/feature_descriptor_g/conv4b/bias:0', \
                          'top/feature_descriptor_g/conv5a/weight:0',  'top/feature_descriptor_g/conv5a/bias:0', \
                          'top/feature_descriptor_g/fc6/weight:0',    'top/feature_descriptor_g/fc6/bias:0', \
                          'top/feature_descriptor_g/fc7/weight:0',    'top/feature_descriptor_g/fc7/bias:0' ]

    feature_a0_VarsList = ['top/feature_descriptor_a0/conv1/weight:0',  'top/feature_descriptor_a0/conv1/bias:0', \
                           'top/feature_descriptor_a0/conv2/weight:0',  'top/feature_descriptor_a0/conv2/bias:0', \
                           'top/feature_descriptor_a0/conv3a/weight:0',  'top/feature_descriptor_a0/conv3a/bias:0', \
                           'top/feature_descriptor_a0/conv4a/weight:0', 'top/feature_descriptor_a0/conv4a/bias:0', \
                           #'top/feature_descriptor_a0/conv4b/weight:0', 'top/feature_descriptor_a0/conv4b/bias:0', \
                           'top/feature_descriptor_a0/conv5a/weight:0',  'top/feature_descriptor_a0/conv5a/bias:0', \
                           'top/feature_descriptor_a0/fc6/weight:0',    'top/feature_descriptor_a0/fc6/bias:0', \
                           'top/feature_descriptor_a0/fc7/weight:0',    'top/feature_descriptor_a0/fc7/bias:0' ]

    feature_a1_VarsList = ['top/feature_descriptor_a1/conv1/weight:0',  'top/feature_descriptor_a1/conv1/bias:0', \
                           'top/feature_descriptor_a1/conv2/weight:0',  'top/feature_descriptor_a1/conv2/bias:0', \
                           'top/feature_descriptor_a1/conv3a/weight:0',  'top/feature_descriptor_a1/conv3a/bias:0', \
                           'top/feature_descriptor_a1/conv4a/weight:0', 'top/feature_descriptor_a1/conv4a/bias:0', \
                           #'top/feature_descriptor_a1/conv4b/weight:0', 'top/feature_descriptor_a1/conv4b/bias:0', \
                           'top/feature_descriptor_a1/conv5a/weight:0',  'top/feature_descriptor_a1/conv5a/bias:0', \
                           'top/feature_descriptor_a1/fc6/weight:0',    'top/feature_descriptor_a1/fc6/bias:0', \
                           'top/feature_descriptor_a1/fc7/weight:0',    'top/feature_descriptor_a1/fc7/bias:0' ]

    classifier_sm_VarsList      = ['top/classifier/sm/weight:0',  'top/classifier/sm/bias:0'] 
    classifier_sm_2f1c_VarsList = ['top/classifier_2f1c/sm/weight:0',  'top/classifier_2f1c/sm/bias:0'] 
    classifier_sm_3f1c_VarsList = ['top/classifier_3f1c/sm/weight:0',  'top/classifier_3f1c/sm/bias:0'] 
    
def getDateTime():
    t = str(datetime.now())
    t = t[5:16]
    t = t.replace(' ','-')
    t = t.replace(':','-')
    return t