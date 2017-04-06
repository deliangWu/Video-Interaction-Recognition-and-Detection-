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
            self._files = np.array([f for f in listdir(path) if isfile(join(path,f)) and re.search('.avi',f) is not None])
            fs = np.array([[sequence(fileName), join(path,fileName), Label(fileName)] for i,fileName in enumerate(self._files)])
            self._filesSet = np.append(self._filesSet,fs,axis = 0)
        
    def splitTrainingTesting(self,n, loadTrainingEn = True):
        testingIndex = [i for i,fileSet in enumerate(self._filesSet) if int(fileSet[0]) == n]
        trainingIndex = [i for i,fileSet in enumerate(self._filesSet) if int(fileSet[0]) != n]
        self._trainingFilesSet = self._filesSet[trainingIndex]
        self._testingFilesSet = self._filesSet[testingIndex]
        if loadTrainingEn == True:
            self.loadTrainingAll()
        return None
    
    def loadTrainingAll(self, shuffleEn = True):
        cnt_file = 0
        for file in self._trainingFilesSet:
            video = vpp.videoProcess(file[1],self._frmSize)
            self._trainingVideos = np.append(self._trainingVideos,video,axis=0)
            labelCode = vpp.int2OneHot(int(file[2]),6)
            label = np.repeat(np.reshape(labelCode,(1,6)),video.shape[0],axis=0)
            self._trainingLabels=  np.append(self._trainingLabels,label,axis=0)
            cnt_file+=1
            if cnt_file % 10 == 0:
                print('Loading training videos: ' + str(int(cnt_file * 100 / self._trainingFilesSet.shape[0])) +'%')
        if shuffleEn == True:
            perm = np.arange(self._trainingVideos.shape[0])
            np.random.shuffle(perm)
            self._trainingVideos = self._trainingVideos[perm]
            self._trainingLabels = self._trainingLabels[perm]
        return None
    
    def getTrainingSet(self):
        return [self._trainingVideos, self._trainingLabels]
    
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
        testLabels = np.empty((0,6),dtype=np.float32)        
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
                testLabels = np.append(testLabels,np.reshape(labelCode,(1,6)),axis=0)
        return (testVideos.transpose(1,0,2,3,4,5),testLabels)    
    
    def getFileList(self):
        return self._files
            
class ut_interaction_atomic:
    def __init__(self,paths,frmSize):
        self._ut_a0 = ut_interaction([paths[0]], frmSize)
        self._ut_a1 = ut_interaction([paths[1]], frmSize)
        fileList0 = self._ut_a0.getFileList()
        fileList1 = self._ut_a1.getFileList()
        self._trainingEpoch = 0
        assert np.array_equal(fileList0,fileList1), 'Error, input videos from two set is mismatch!'
        self._trainingPointer = 0
    
    def splitTrainingTesting(self,n):
        self._ut_a0.splitTrainingTesting(n,loadTrainingEn=False)
        self._ut_a1.splitTrainingTesting(n,loadTrainingEn=False)
        self.loadTrainingAll()
        return None
    
    def loadTrainingAll(self):
        self._ut_a0.loadTrainingAll(shuffleEn=False)
        self._trainingSet_a0 = self._ut_a0.getTrainingSet()
        self._ut_a1.loadTrainingAll(shuffleEn=False)
        self._trainingSet_a1 = self._ut_a1.getTrainingSet()
        assert self._trainingSet_a0[0].shape == self._trainingSet_a1[0].shape, 'Error, the shape of two set is mismatch!'
        assert np.array_equal(self._trainingSet_a0[1],self._trainingSet_a1[1]), 'Error, the lable of two set is mismatch!'
        perm = np.arange(self._trainingSet_a0[0].shape[0])
        np.random.shuffle(perm)
        self._trainingSet_a0 = [self._trainingSet_a0[0][perm],self._trainingSet_a0[1][perm]]
        self._trainingSet_a1 = [self._trainingSet_a1[0][perm],self._trainingSet_a1[1][perm]]
        return None
    
    def loadTrainingBatch(self,batch = 16):
        if self._trainingPointer + batch > self._trainingSet_a0[0].shape[0]:
            start = 0
            self._trainingEpoch += 1
            self._trainingPointer = batch
            perm = np.arange(self._trainingSet_a0[0].shape[0])
            np.random.shuffle(perm)
            self._trainingSet_a0 = [self._trainingSet_a0[0][perm],self._trainingSet_a0[1][perm]]
            self._trainingSet_a1 = [self._trainingSet_a1[0][perm],self._trainingSet_a1[1][perm]]
        else:
            start = self._trainingPointer
            self._trainingPointer += batch
        end = self._trainingPointer
        return(self._trainingSet_a0[0][start:end],self._trainingSet_a1[0][start:end],self._trainingSet_a0[1][start:end])
        
    def loadTesting(self):
        [testVideos_a0, testLables_a0] = self._ut_a0.loadTesting()
        [testVideos_a1, testLables_a1] = self._ut_a1.loadTesting()
        assert testVideos_a0.shape == testVideos_a1.shape, 'Error, the video shape between two set is mismatch!'
        assert np.array_equal(testLables_a0, testLables_a1), "Error, the lable between two set is mismatch!"
        return(testVideos_a0, testVideos_a1, testLables_a0)
    
class ut_interaction_ga:
    def __init__(self,paths,frmSize):
        self._ut_g  = ut_interaction([paths[0]], frmSize[0])
        self._ut_a0 = ut_interaction([paths[1]], frmSize[1])
        self._ut_a1 = ut_interaction([paths[2]], frmSize[1])
        fileList_g  = self._ut_g.getFileList()
        fileList_a0 = self._ut_a0.getFileList()
        fileList_a1 = self._ut_a1.getFileList()
        self._trainingEpoch = 0
        self._trainingPointer = 0
        assert np.array_equal(fileList_a0,fileList_a1) and np.array_equal(fileList_g, fileList_a0), 'Error, input videos from three sets are mis-match!'
    
    def splitTrainingTesting(self,n):
        self._ut_g.splitTrainingTesting(n,loadTrainingEn=False)
        self._ut_a0.splitTrainingTesting(n,loadTrainingEn=False)
        self._ut_a1.splitTrainingTesting(n,loadTrainingEn=False)
        self.loadTrainingAll()
        return None
    
    def loadTrainingAll(self):
        self._ut_g.loadTrainingAll(shuffleEn=False)
        self._trainingSet_g = self._ut_g.getTrainingSet()
        self._ut_a0.loadTrainingAll(shuffleEn=False)
        self._trainingSet_a0 = self._ut_a0.getTrainingSet()
        self._ut_a1.loadTrainingAll(shuffleEn=False)
        self._trainingSet_a1 = self._ut_a1.getTrainingSet()
        #assert self._trainingSet_g[0].shape  == self._trainingSet_a0[0].shape and \
        #       self._trainingSet_a0[0].shape == self._trainingSet_a1[0].shape, \
        #       'Error, the shapes of three sets are mismatch!'
        assert np.array_equal(self._trainingSet_g[1], self._trainingSet_a0[1]) and \
               np.array_equal(self._trainingSet_a0[1],self._trainingSet_a1[1]), \
               'Error, the lables of three sets are mismatch!'
        perm = np.arange(self._trainingSet_g[0].shape[0])
        np.random.shuffle(perm)
        self._trainingSet_g  = [self._trainingSet_g[0][perm] ,self._trainingSet_g[1][perm]]
        self._trainingSet_a0 = [self._trainingSet_a0[0][perm],self._trainingSet_a0[1][perm]]
        self._trainingSet_a1 = [self._trainingSet_a1[0][perm],self._trainingSet_a1[1][perm]]
        return None
    
    def loadTrainingBatch(self,batch = 16):
        if self._trainingPointer + batch > self._trainingSet_a0[0].shape[0]:
            start = 0
            self._trainingEpoch += 1
            self._trainingPointer = batch
            perm = np.arange(self._trainingSet_a0[0].shape[0])
            np.random.shuffle(perm)
            self._trainingSet_g  = [self._trainingSet_g[0][perm], self._trainingSet_g[1][perm]]
            self._trainingSet_a0 = [self._trainingSet_a0[0][perm],self._trainingSet_a0[1][perm]]
            self._trainingSet_a1 = [self._trainingSet_a1[0][perm],self._trainingSet_a1[1][perm]]
        else:
            start = self._trainingPointer
            self._trainingPointer += batch
        end = self._trainingPointer
        return(self._trainingSet_g[0][start:end],self._trainingSet_a0[0][start:end],self._trainingSet_a1[0][start:end],self._trainingSet_a0[1][start:end])
        
    def loadTesting(self):
        [testVideos_g,  testLables_g]  = self._ut_g.loadTesting()
        [testVideos_a0, testLables_a0] = self._ut_a0.loadTesting()
        [testVideos_a1, testLables_a1] = self._ut_a1.loadTesting()
        #assert testVideos_g.shape == testVideos_a0.shape and \
        #       testVideos_a0.shape == testVideos_a1.shape, \
        #       'Error, the video shape between three sets are mismatch!'
        assert np.array_equal(testLables_g, testLables_a0) and \
               np.array_equal(testLables_a0, testLables_a1), \
               "Error, the lable between three sets are mismatch!"
        return(testVideos_g, testVideos_a0, testVideos_a1, testLables_a0)
        

class ut_interaction_set1(ut_interaction):
    def __init__(self,frmSize):
        path = [common.path.utSet1Path]
        ut_interaction.__init__(self,path,frmSize)

class ut_interaction_set1_a(ut_interaction):
    def __init__(self,frmSize):
        paths = [common.path.utSet1_a0_Path,common.path.utSet1_a1_Path]
        ut_interaction.__init__(self,paths,frmSize)

class ut_interaction_set1_atomic(ut_interaction_atomic):
    def __init__(self,frmSize):
        paths = [common.path.utSet1_a0_Path,common.path.utSet1_a1_Path]
        ut_interaction_atomic.__init__(self,paths,frmSize)

class ut_interaction_set1_ga(ut_interaction_ga):
    def __init__(self,frmSize):
        paths = [common.path.utSet1Path, common.path.utSet1_a0_Path,common.path.utSet1_a1_Path]
        ut_interaction_ga.__init__(self,paths,frmSize)

class ut_interaction_set2(ut_interaction):
    def __init__(self,frmSize):
        path = [common.path.utSet2Path]
        ut_interaction.__init__(self,path,frmSize)

class ut_interaction_set2_atomic(ut_interaction_atomic):
    def __init__(self,frmSize):
        paths = [common.path.utSet2_a0_Path,common.path.utSet2_a1_Path]
        ut_interaction_atomic.__init__(self,paths,frmSize)

class ut_interaction_set2_a(ut_interaction):
    def __init__(self,frmSize):
        paths = [common.path.utSet2_a0_Path,common.path.utSet2_a1_Path]
        ut_interaction.__init__(self,paths,frmSize)




# for detection task
from mmap import mmap, ACCESS_READ
from xlrd import open_workbook
import cv2

def labelToString(label):
    activitys = ['Hand Shaking', 'Hugging', 'Kicking', 'Pointing', 'Punching', 'Pushing']
    return activitys[label]

def getGroundTruth(setNo, seqNo):
    workbook = open_workbook('D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_labels_110912.xls')
    groundTruth = []
    for sheet in workbook.sheets():
        if sheet.name == 'Set'+str(setNo):
            for row in range(62):
                line = []
                if sheet.cell(row,0).value == 'seq' + str(seqNo):
                    for col in range(1,8):
                        line.append(int(sheet.cell(row,col).value))
                    groundTruth.append(line)
    return np.array(groundTruth)
                         

if __name__ == '__main__':
    setNo,seqNo = 2,18
    videoName = 'D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_set' + str(setNo) + '/seq' + str(seqNo) +'.avi'
    gt = getGroundTruth(setNo, seqNo)
    print(gt)
    cv2.namedWindow('video player')    
    cap = cv2.VideoCapture(videoName)
    ret,frame = cap.read()
    frmNo = 0
    gt_i = 0
    while(ret):
        if gt_i < gt.shape[0] :
            if frmNo > gt[gt_i][1] and frmNo < gt[gt_i][2]:
                cv2.rectangle(frame, (gt[gt_i][3], gt[gt_i][4]), (gt[gt_i][5], gt[gt_i][6]), (gt_i * 40, 255 - gt_i * 40, gt_i * 40), thickness=1)
                cv2.putText(frame, labelToString(gt[gt_i][0]), (gt[gt_i][3],gt[gt_i][4] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            if frmNo > gt[gt_i][2]:
                gt_i+=1
        cv2.imshow('video player', frame)
        ret,frame = cap.read()
        frmNo +=1
        if cv2.waitKey(30) == 27:
            break
        
        
