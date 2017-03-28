from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import videoPreProcess as vpp
import sys
sys.path.insert(1,'../common')
import common

def sequence(fileName):
    return int(fileName[fileName.index("_") + 1 : fileName.rindex("_")])

def Label(fileName):
    return int(fileName[fileName.index(".") - 1 : fileName.index(".")])

class ut_interaction:
    def __init__(self,paths,frmSize):
        self._frmSize = frmSize
        self._filesSet = np.empty((0,3)) 
        self._trainingFilesSet= []
        self._testingFilesSet= []
        self._trainingVideos = np.empty((0,16) + self._frmSize,dtype = np.uint8)
        self._trainingLabels = np.empty((0,6),dtype=np.float32)
        self._trainingPointer = 0
        self._trainingEpoch = 0
        for path in paths:
            files = np.array([f for f in listdir(path) if isfile(join(path,f)) and re.search('.avi',f) is not None])
            fs = np.array([[sequence(fileName), join(path,fileName), Label(fileName)] for i,fileName in enumerate(files)])
            self._filesSet = np.append(self._filesSet,fs,axis = 0)
        
    def splitTrainingTesting(self,n):
        testingIndex = [i for i,fileSet in enumerate(self._filesSet) if int(fileSet[0]) == n]
        trainingIndex = [i for i,fileSet in enumerate(self._filesSet) if int(fileSet[0]) != n]
        self._trainingFilesSet = self._filesSet[trainingIndex]
        self._testingFilesSet = self._filesSet[testingIndex]
        self.loadTrainingAll()
        return None
    
    def loadTrainingAll(self):
        for file in self._trainingFilesSet:
            video = vpp.videoProcess(file[1],self._frmSize)
            self._trainingVideos = np.append(self._trainingVideos,video,axis=0)
            labelCode = vpp.int2OneHot(int(file[2]),6)
            label = np.repeat(np.reshape(labelCode,(1,6)),video.shape[0],axis=0)
            self._trainingLabels=  np.append(self._trainingLabels,label,axis=0)
            print('.',end='')
        perm = np.arange(self._trainingVideos.shape[0])
        np.random.shuffle(perm)
        self._trainingVideos = self._trainingVideos[perm]
        self._trainingLabels = self._trainingLabels[perm]
        return None
    
    def loadTrainingBatch(self,batch = 16):
        if self._trainingPointer + batch >= self._trainingVideos.shape[0]:
            start = 0
            self._trainingEpoch += 1
            self._trainingPointer = batch
            perm = np.arange(self._trainingVideos.shape[0])
            np.random.shuffle(perm)
            self._trainingVideos = self._trainingVideos[perm]
            self._trainingLabels = self._trainingLabels[perm]
        else:
            start = self._trainingPointer
            self._trainingPointer += batch
        end = self._trainingPointer
        return(self._trainingVideos[start:end],self._trainingLabels[start:end])
    
    def loadTesting(self):
        testVideos = np.empty((0,3,16) + self._frmSize, dtype=np.uint8)        
        testlabels = np.empty((0,6),dtype=np.float32)        
        for file in self._testingFilesSet:
            labelCode = vpp.int2OneHot(int(file[2]),6)
            video = vpp.videoProcess(file[1],self._frmSize,RLFlipEn=False,NormEn=True)
            if video is not None:
                numOfClips = video.shape[0]
                if numOfClips == 1:
                    index = [0,0,0]
                elif numOfClips == 2:
                    index = [0,1,1]
                else:
                    index = range(int(numOfClips/2) - 1, int(numOfClips/2) + 2)
                video = video[index]
                video = np.reshape(video,(1,) + video.shape)
                testVideos = np.append(testVideos,video,axis=0)
                testlabels = np.append(testlabels,np.reshape(labelCode,(1,6)),axis=0)
        return (testVideos.transpose(1,0,2,3,4,5),testlabels)    
            
            
            
                
class ut_interaction_set1(ut_interaction):
    def __init__(self,frmSize):
        path = [common.path.utSet1Path]
        ut_interaction.__init__(self,path,frmSize)

class ut_interaction_set1_a(ut_interaction):
    def __init__(self,frmSize):
        paths = [common.path.utSet1_a0_Path,common.path.utSet1_a1_Path]
        ut_interaction.__init__(self,paths,frmSize)

class ut_interaction_set2(ut_interaction):
    def __init__(self,frmSize):
        path = [common.path.utSet2Path]
        ut_interaction.__init__(self,path,frmSize)

class ut_interaction_set2_a(ut_interaction):
    def __init__(self,frmSize):
        paths = [common.path.utSet2_a0_Path,common.path.utSet2_a1_Path]
        ut_interaction.__init__(self,paths,frmSize)
        

if __name__ == '__main__':
    utset = ut_interaction_set2_a((112,80,3))
    seq_bias = 10 
    for seq in range(seq_bias+1,seq_bias+11):
        print('**************************************************************')
        print('current sequence is ', seq)
        print('**************************************************************')
        utset.splitTrainingTesting(seq)
        train = utset.loadTrainingBatch(16)
        test = utset.loadTesting()
        for v in test[0][:,0]:
            print(v.shape)
            vpp.videoPlay(v)
        print(test[0].shape)
        print('    ')
        
        