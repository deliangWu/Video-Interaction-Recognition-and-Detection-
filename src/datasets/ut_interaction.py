import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import videoPreProcess


def sequence(fileName):
    return int(fileName[fileName.index("_") + 1 : fileName.rindex("_")])

def Label(fileName):
    return int(fileName[fileName.index(".") - 1 : fileName.index(".")])

class ut_interaction:
    def __init__(self,path,frmSize):
        self._path = path
        self._frmSize = frmSize
        self._trainingSet = []
        self._testingSet= []
        self._videos = np.empty((0,16) + self._frmSize,dtype = np.uint8)
        self._labels = np.empty((0,6),dtype=np.float32)
        self._seqs = np.empty((0),dtype=np.int16)
        self._labs = np.empty((0),dtype=np.int16)
        self._trainingIndex = []
        self._trainingPointer = 0
        self._trainingEpoch = 0
        files = np.array([f for f in listdir(path) if isfile(join(path,f)) and re.search('.avi',f) is not None])
        self._filesSet = np.array([[sequence(fileName), fileName, Label(fileName)] for fileName in files])
        for file in self._filesSet:
            print(file[1])
            video = videoPreProcess.videoProcess(join(path,file[1]),self._frmSize)
            self._videos = np.append(self._videos,video,axis=0)
            labelCode = videoPreProcess.int2OneHot(int(file[2]),6)
            label = np.repeat(np.reshape(labelCode,(1,6)),video.shape[0],axis=0)
            self._labels =  np.append(self._labels,label,axis=0)
            self._seqs = np.append(self._seqs,np.repeat(int(file[0]),video.shape[0]))
            self._labs = np.append(self._labs,np.repeat(int(file[2]),video.shape[0]))
        
    def splitTrainingTesting(self,n):
        testingIndex = [i for i,j in enumerate(self._seqs) if int(j) == n]
        trainingIndex = [i for i,j in enumerate(self._seqs) if int(j) != n]
        self._trainingSet = [self._videos[trainingIndex], self._labels[trainingIndex]]
        self._testingSet = [self._videos[testingIndex], self._labels[testingIndex],self._seqs[testingIndex], self._labs[testingIndex]]
        self._trainingIndex = np.arange(len(trainingIndex))
        return None
    
    def loadTraining(self,batch = 16):
        if self._trainingPointer + batch >= len(self._trainingIndex):
            np.random.shuffle(self._trainingIndex)
            start = 0
            self._trainingEpoch += 1
            self._trainingPointer = batch
        else:
            start = self._trainingPointer
            self._trainingPointer += batch
        end = self._trainingPointer
        return(self._trainingSet[0][start:end],self._trainingSet[1][start:end])
    
    def loadTesting(self):
        testingIndex = np.arange(len(self._testingSet[0]))
        np.random.shuffle(testingIndex)
        print(self._testingSet[2][testingIndex])
        print(self._testingSet[3][testingIndex])
        return([t[testingIndex] for t in self._testingSet])
        
    def loadTesting_new(self):
        #testingIndex = np.arange(len(self._testingSet[0]))
        #np.random.shuffle(testingIndex)
        testingLabs = self._testingSet[3]
        for i in range(6):
            iLoc = [l for l,v in enumerate(testingLabs) if v == i]
            if iLoc == []:
                continue
            numOfClips = int(len(iLoc)/2)
            if numOfClips == 1:
                index = [iLoc[0],iLoc[0],iLoc[0]]
            elif numOfClips == 2:
                index = [iLoc[0],iLoc[1],iLoc[1]]
            else:
                index = iLoc[int(numOfClips/2) - 1: int(numOfClips/2) + 2]
            videosI = self._testingSet[0][index]
            labelI = self._testingSet[1][iLoc[0]]
            if videosI is not None:
                if i == 0:
                    videos = [videosI]
                    labels = [labelI]
                else:
                    videos.append(videosI)
                    labels.append(labelI)
        return(np.array(videos).transpose(1,0,2,3,4,5),np.array(labels))
            
            
            
                
class ut_interaction_set1(ut_interaction):
    def __init__(self,frmSize):
        if os.name == 'nt':
            path = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1"
        else:
            path = "/home/wdl/3DCNN/datasets/ut_interaction/segmented_set1"
        ut_interaction.__init__(self,path,frmSize)

class ut_interaction_set2(ut_interaction):
    def __init__(self,frmSize):
        if os.name == 'nt':
            path = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2"
        else:
            path = "/home/wdl/3DCNN/datasets/ut_interaction/segmented_set2"
        ut_interaction.__init__(self,path,frmSize)
        

if __name__ == '__main__':
    set1 = ut_interaction_set1((112,144,3))
    
    for seq in range(1,11):
        print '**************************************************************'
        print 'current sequence is ', seq
        print '**************************************************************'
        set1.splitTrainingTesting(seq)
        test = set1.loadTesting_new()
        print test[0].shape
        for clips in test[0]:
            for v in clips:
                videoPreProcess.videoPlay(v,25)
        print test[1]
        print test[1].shape
        print '    '
        
        