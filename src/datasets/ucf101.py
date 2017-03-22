import cv2
import numpy as np
import random
import videoPreProcess
import time

def videoProcess(fileName,frmSize):
    v1 = videoPreProcess.videoRead(fileName)
    if v1 is not None:
        v2 = videoPreProcess.videoRezise(v1,frmSize)
        v3 = videoPreProcess.videoSimplify(v2)
        v4 = videoPreProcess.downSampling(v3)
        v5 = videoPreProcess.batchFormat(v4)
        return v5
    else:
        return None

def int2OneHot(din,range):
    code = np.zeros(range,dtype=np.float32)
    code[din-1] = 1
    return code
    
class ucf101:
    def __init__(self,frmSize,numOfClasses):
        self._datasetPath = "/home/wdl/3DCNN/datasets/UCF101/"
        self._frmSize = frmSize
        self._numOfClasses = numOfClasses
        self._validLabels = range(1,numOfClasses+1)
        self._classInd = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt",dtype=bytes).astype(str)
        
        self._trainFilelist1 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt",dtype=bytes).astype(str)
        self._trainFilelist1 = self._trainFilelist1
        print(self._trainFilelist1.shape)
        np.random.shuffle(self._trainFilelist1)
        
        #self._trainFilelist2 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._trainFilelist2)
        
        #self._trainFilelist3 = np.loadtxt(self._datasetPath + "/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._trainFilelist3)
        
        self._testFilelist1 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt",dtype=bytes).astype(str)
        self._testFilelist1 = self._testFilelist1
        print(self._testFilelist1.shape)
        np.random.shuffle(self._testFilelist1)
        
        #self._testFilelist2 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._testFilelist2)
        
        #self._testFilelist3 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._testFilelist3)
        self.trainFileIndex = 0
        
        
    def loadTest(self,n = 0):
        testVideos = np.empty((0,16) + self._frmSize + (3,),dtype=np.uint8)        
        testlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        for file in self._testFilelist1:
            label = self._classInd[:,0][np.where(self._classInd == file[0:file.index("/")])[0]]
            if int(label[0]) in self._validLabels:
                print(file,'--------- label is ',label[0])
                labelCode = int2OneHot(int(label[0]),self._numOfClasses)
                fileName = self._datasetPath + 'UCF-101/' + file
                video = videoProcess(fileName,self._frmSize)
                if video is not None:
                    videoLabel = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
                    testVideos = np.append(testVideos,video,axis=0)
                    testlabels = np.append(testlabels,videoLabel,axis=0)
                    if (n > 0) and (testVideos.shape[0] >= n):
                        break
        print ('the shape of testVideos is ',testVideos.shape)
        return (testVideos[0:n],testlabels[0:n])    
    
        
    def loadTrainBatch(self,n):
        trainVideos = np.empty((0,16) + self._frmSize + (3,),dtype=np.uint8)        
        trainlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        for file,label in self._trainFilelist1[self.trainFileIndex:]:
            if int(label) in self._validLabels:
                #print(file,'--------- label is ',label)
                labelCode = int2OneHot(int(label),self._numOfClasses)
                fileName = self._datasetPath + 'UCF-101/' + file
                video = videoProcess(fileName,self._frmSize)
                if video is not None:
                    videoLabel = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
                    trainVideos = np.append(trainVideos,video,axis=0)
                    trainlabels = np.append(trainlabels,videoLabel,axis=0)
                    if (trainVideos.shape[0] >= n):
                        break
            self.trainFileIndex= self.trainFileIndex+ 1
            if (self.trainFileIndex >= (self._trainFilelist1.shape[0] - 1)):
                self.trainFileIndex= 0
                np.random.shuffle(self._trainFilelist1)
                
        return (trainVideos[0:n],trainlabels[0:n])