from __future__ import print_function
import numpy as np
import random
import os
import time
import multiprocessing as mp
import videoPreProcess as vpp
import videoPreProcess as vpp
import sys
sys.path.insert(1,'../common')
import common

def validSet(filelist,validLabels):
    tmpList = np.empty((0,2))
    for file,label in filelist:
        if int(label) in validLabels:
            tmpList = np.append(tmpList,np.reshape(np.array([file,label]),(1,2)),axis=0)
    return(tmpList)
    

class ucf101:
    def __init__(self,frmSize,numOfClasses):
        self._datasetPath = common.path.ucfPath
        self._frmSize = frmSize
        self._numOfClasses = numOfClasses
        self._validLabels = range(1,numOfClasses+1)
        self._classInd = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt",dtype=bytes).astype(str)
        
        self._trainFilelist1 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt",dtype=bytes).astype(str)
        np.random.shuffle(self._trainFilelist1)
        self._trainFilelist1 = validSet(self._trainFilelist1,self._validLabels)
        
        self._trainVideos = np.empty((0,16) + self._frmSize, dtype=np.uint8)        
        self._trainlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        self._numOfTrainSamples = 0
        self._trainFileIndex = 0
        self._trainEpoch = 0
        
        #self._trainFilelist2 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._trainFilelist2)
        
        #self._trainFilelist3 = np.loadtxt(self._datasetPath + "/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._trainFilelist3)
        
        self._testFilelist1 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt",dtype=bytes).astype(str)
        np.random.shuffle(self._testFilelist1)
        
        #self._testFilelist2 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._testFilelist2)
        
        #self._testFilelist3 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._testFilelist3)
        
        
    def loadTest(self,n = 0):
        testVideos = np.empty((0,3,16) + self._frmSize, dtype=np.uint8)        
        testlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        for file in self._testFilelist1:
            label = self._classInd[:,0][np.where(self._classInd == file[0:file.index("/")])[0]]
            if int(label[0]) in self._validLabels:
                labelCode = vpp.int2OneHot(int(label[0])-1,self._numOfClasses)
                fileName = self._datasetPath + 'UCF-101/' + file
                video = vpp.videoProcess(fileName,self._frmSize,RLFlipEn=False,NormEn=True)
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
                    testlabels = np.append(testlabels,np.reshape(labelCode,(1,self._numOfClasses)),axis=0)
                    if (n > 0) and (testVideos.shape[0] >= n):
                        break
        return (testVideos.transpose(1,0,2,3,4,5),testlabels)    
    
        
    def loadTrainBatch(self,n):
        if (self._trainFileIndex + n > self._trainVideos.shape[0]):
            start = 0
            self._trainFileIndex = n
            self._trainEpoch += 1
            # shuffle the data
            perm = np.arange(self._numOfTrainSamples)
            np.random.shuffle(perm)
            self._trainVideos = self._trainVideos[perm]
            self._trainlabels = self._trainlabels[perm]
            print('current epoch is ',self._trainEpoch)
        else:
            start = self._trainFileIndex
            self._trainFileIndex += n
        
        end = self._trainFileIndex
        return self._trainVideos[start:end],self._trainlabels[start:end]
        
    def loadTrainAll(self,n=0):
        cntVideos = 0
        for file,label in self._trainFilelist1:
            labelCode = vpp.int2OneHot(int(label)-1,self._numOfClasses)
            fileName = self._datasetPath + 'UCF-101/' + file
            video = vpp.videoProcess(fileName,self._frmSize,RLFlipEn=False)
            if video is not None:
                cntVideos += 1
                videoLabel = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
                self._trainVideos = np.append(self._trainVideos,video,axis=0)
                self._trainlabels = np.append(self._trainlabels,videoLabel,axis=0)
                if n > 0 and (cntVideos >= n):
                    break
                if cntVideos%1 == 0:
                    print(cntVideos,' files are loaded!')
        self._numOfTrainSamples = self._trainVideos.shape[0]
        print('training videos are loaded, the shape of loaded videos is ',self._numOfTrainSamples)
        return None
    
    def loadTrainBatchMP(self,n):
        if self._trainFileIndex + n > self._trainFilelist1.shape[0] - 1:
            start = 0
            self._trainFileIndex = n
            self._trainEpoch += 1
            # shuffle the data
            np.random.shuffle(self._trainFilelist1)
            print('current epoch is ',self._trainEpoch)
        else:
            start = self._trainFileIndex
            self._trainFileIndex += n
        end = self._trainFileIndex
        
        trainVideos = np.empty((0,16) + self._frmSize, dtype=np.uint8)        
        trainlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        for file,label in self._trainFilelist1[start:end]:
            labelCode = vpp.int2OneHot(int(label)-1,self._numOfClasses)
            fileName = self._datasetPath + 'UCF-101/' + file
            video = vpp.videoProcess(fileName,self._frmSize,RLFlipEn=False)
            if video is not None:
                videoLabel = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
                trainVideos = np.append(trainVideos,video,axis=0)
                trainlabels = np.append(trainlabels,videoLabel,axis=0)
        return [trainVideos,trainlabels]
    
    def loadTrainAllMP(self,mp=(1,0),q):
        videosPerProcess = int(self._trainFilelist1.shape[0]/mp[0])
        if mp[1] >= mp[0]:
            trainFilelist = self._trainFilelist1[mp[1] * videosPerProcess:]
        else:
            trainFilelist = self._trainFilelist1[mp[1] * videosPerProcess:(mp[1] + 1)*videosPerProcess]
        cntVideos = 0
        trainVideos = np.empty((0,16) + self._frmSize, dtype=np.uint8)        
        trainlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        for file,label in trainFilelist:
            labelCode = vpp.int2OneHot(int(label)-1,self._numOfClasses)
            fileName = self._datasetPath + 'UCF-101/' + file
            video = vpp.videoProcess(fileName,self._frmSize,RLFlipEn=False)
            if video is not None:
                videoLabel = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
                trainVideos = np.append(trainVideos,video,axis=0)
                trainlabels = np.append(trainlabels,videoLabel,axis=0)
                if cntVideos%3 == 0:
                    print(cntVideos,' files are loaded!')
                cntVideos += 1
        q.put([trainVideos,trainlabels])
    
    def runloadTrainAllMP(self,numOfProcesses):
        processes = []
        queues=[]
        for i in range(numOfProcesses):
            q = mp.Queue()
            p = mp.Process(target=self.loadTrainAllMP,args=((numOfProcesses,i),q,))
            processes.append(p)
            queues.append(q)
        [x.start() for x in processes]
        [x.join() for x in processes]
        trainSet = [x.get() for x in queues]
        for ts in trainSet:
            print(ts[1].shape)
        
if __name__ == '__main__':
    frmSize = (112,80,3)
    numOfClasses =int(sys.argv[2])
    ucf = ucf101(frmSize, numOfClasses)    
    numOfProcesses = int(sys.argv[1])
    print('start time is ',time.ctime())
    ucf.runloadTrainAllMP(numOfProcesses)
    print('end time is ',time.ctime())
    
    
    