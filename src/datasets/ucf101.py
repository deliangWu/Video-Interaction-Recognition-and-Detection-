from __future__ import print_function
import numpy as np
import random
import os
import time
import multiprocessing as mp
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

def genTestFile(filelist,classInd,validLabels):
    tmpList = np.empty((0,2))
    for file in filelist:
        label = classInd[:,0][np.where(classInd == file[0:file.index("/")])[0]]
        if int(label[0]) in validLabels:
            tmpList = np.append(tmpList,np.reshape(np.array([file,int(label[0])]),(1,2)),axis=0)
    return(tmpList)
            
class ucf101:
    def __init__(self,frmSize,numOfClasses):
        self._datasetPath = common.path.ucfPath
        self._frmSize = frmSize
        self._numOfClasses = numOfClasses
        validLabels = range(1,numOfClasses+1)
        
        self._trainFilelist1 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt",dtype=bytes).astype(str)
        self._trainFilelist1 = validSet(self._trainFilelist1,validLabels)
        np.random.shuffle(self._trainFilelist1)
        
        self._trainVideos = np.empty((0,16) + self._frmSize, dtype=np.uint8)        
        self._trainlabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        self._trainFileIndex = 0
        self._trainEpoch = 0
        
        #self._trainFilelist2 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._trainFilelist2)
        
        #self._trainFilelist3 = np.loadtxt(self._datasetPath + "/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._trainFilelist3)
        
        self._testFilelist1 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt",dtype=bytes).astype(str)
        classInd = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt",dtype=bytes).astype(str)
        self._testFilelist1 = genTestFile(self._testFilelist1,classInd,validLabels)        
        np.random.shuffle(self._testFilelist1)
        
        #self._testFilelist2 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._testFilelist2)
        
        #self._testFilelist3 = np.loadtxt(self._datasetPath + "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt",dtype=bytes).astype(str)
        #np.random.shuffle(self._testFilelist3)
        
    def loadVideosAllMP(self,filelist,q,mp=(1,0),videoType='train'):
        videosPerProcess = int(filelist.shape[0]/mp[0])
        if mp[1] >= mp[0] - 1:
            subFilelist = filelist[mp[1] * videosPerProcess:]
        else:
            subFilelist = filelist[mp[1] * videosPerProcess:(mp[1] + 1)*videosPerProcess]
        cntVideos = 0
        numOfVideos = subFilelist.shape[0]
        videos = np.empty((0,16) + self._frmSize, dtype=np.uint8)        
        labels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        for file,label in subFilelist:
            labelCode = vpp.int2OneHot(int(label)-1,self._numOfClasses)
            fileName = self._datasetPath + 'UCF-101/' + file
            video = vpp.videoProcess(fileName,self._frmSize,RLFlipEn=False,NormEn=(videoType=='test'))
            if video is not None and video.shape[0] >= 1:
                if videoType == 'test':
                    numOfClips = video.shape[0]
                    if numOfClips == 1:
                        index = [0,0,0]
                    elif numOfClips == 2:
                        index = [0,1,1]
                    else:
                        index = range(int(numOfClips/2) - 1, int(numOfClips/2) + 2)
                    video = video[index]
                videoLabel = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
                videos = np.append(videos,video,axis=0)
                labels = np.append(labels,videoLabel,axis=0)
                cntVideos += 1
                if cntVideos%max(1,int(videosPerProcess/20)) == 0:
                    print('Process-' + str(os.getpid()) + ' Loading videos for ' + videoType +' : ' + str(int(float(cntVideos * 100) / videosPerProcess)) +'%')
                    
        print('Process-' + str(os.getpid()) + ' finished!')
        q.put([videos,labels])
    
    def runloadVideosAllMP(self,filelist,numOfProcesses=8,videoType = 'train'):
        videos = np.empty((0,16) + self._frmSize, dtype=np.uint8)        
        labels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        videosSet = [videos, labels]
        processes = []
        queues=[]
        
        for i in range(numOfProcesses):
            q = mp.Queue()
            p = mp.Process(target=self.loadVideosAllMP,args=(filelist, q, (numOfProcesses,i),videoType,))
            processes.append(p)
            queues.append(q)
        
        [x.start() for x in processes]
        cnt = 0
        while True:
            for qg in queues:
                subVideosSet = qg.get()
                cnt += 1
                videosSet = [np.append(i,j,0) for i,j in zip(videosSet,subVideosSet)]
            if cnt >= numOfProcesses:
                break
        [x.join() for x in processes]
        print('Dataset UCF101 for ' + videoType + ' is ready!')
        return videosSet
    
    def loadTrainingAll(self,numOfProcesses):
        videos,labels = self.runloadVideosAllMP(self._trainFilelist1,numOfProcesses=numOfProcesses,videoType='train')
        perm = np.arange(videos.shape[0])
        np.random.shuffle(perm)
        self._trainVideos = videos[perm]
        self._trainlabels = labels[perm]
        return None
        
    def loadTesting(self,numOfProcesses):
        testingVideos, testingLabels = self.runloadVideosAllMP(self._testFilelist1,numOfProcesses=numOfProcesses, videoType='test')
        testingVideos = np.reshape(testingVideos,(int(testingVideos.shape[0]/3),3)+testingVideos.shape[1:])
        testingVideos = testingVideos.transpose(1,0,2,3,4,5)
        testingLabels = np.reshape(testingLabels,(int(testingLabels.shape[0]/3),3)+testingLabels.shape[1:])
        testingLabels = testingLabels.transpose(1,0,2)
        assert np.array_equal(testingLabels[0], testingLabels[1]) and np.array_equal(testingLabels[1], testingLabels[2]), 'Error!'
        return [testingVideos,testingLabels[0]]
    
    def loadTrainBatch(self,n):
        if (self._trainFileIndex + n > self._trainVideos.shape[0]):
            start = 0
            self._trainFileIndex = n
            self._trainEpoch += 1
            # shuffle the data
            perm = np.arange(self._trainVideos.shape[0])
            np.random.shuffle(perm)
            self._trainVideos = self._trainVideos[perm]
            self._trainlabels = self._trainlabels[perm]
            print('current epoch is ',self._trainEpoch)
        else:
            start = self._trainFileIndex
            self._trainFileIndex += n
        
        end = self._trainFileIndex
        return self._trainVideos[start:end],self._trainlabels[start:end]

if __name__ == '__main__':
    frmSize = (112,80,3)
    numOfClasses =int(sys.argv[2])
    ucf = ucf101(frmSize, numOfClasses)    
    numOfProcesses = int(sys.argv[1])
    tv,tl = ucf.loadTesting(numOfProcesses)
    #ucf.loadTrainingAll(numOfProcesses)
    #tbv,tbl = ucf.loadTrainBatch(20)
    #print(tbv.shape, '++++++++++++++++',tbl.shape)
    #print(tv.shape,' ------------- ',tl.shape)
    #print(tl)
    #for x in tbv:
    #    vpp.videoPlay(x,fps=10)
    #print('start time is ',time.ctime())
    #ucf.runloadTrainAllMP(numOfProcesses)
    #print('end time is ',time.ctime()) 
    