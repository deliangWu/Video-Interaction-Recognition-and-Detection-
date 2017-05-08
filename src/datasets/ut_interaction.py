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
    def __init__(self,paths,frmSize,numOfClasses = 6):
        self._frmSize = frmSize
        self._filesSet = np.empty((0,3)) 
        self._trainingFilesSet= []
        self._testingFilesSet= []
        for path in paths:
            self._files = np.array([f for f in listdir(path) if isfile(join(path,f)) and \
                                                                re.search('.avi',f) is not None and \
                                                                Label(f) < numOfClasses])
            fs = np.array([[sequence(fileName), join(path,fileName), Label(fileName)] for i,fileName in enumerate(self._files)])
            self._filesSet = np.append(self._filesSet,fs,axis = 0)
        
    def splitTrainingTesting(self,n, loadTrainingEn = False):
        testingIndex = [i for i,fileSet in enumerate(self._filesSet) if int(fileSet[0]) == n]
        trainingIndex = [i for i,fileSet in enumerate(self._filesSet) if int(fileSet[0]) != n]
        self._trainingFilesSet = self._filesSet[trainingIndex]
        self._testingFilesSet = self._filesSet[testingIndex]
        # clean training videos 
        self._trainingVideos = np.empty((0,16) + self._frmSize,dtype = np.uint8)
        self._trainingLabels = np.empty((0,1),dtype=np.float32)
        self._trainingPointer = 0
        self._trainingEpoch = 0
        if loadTrainingEn == True:
            self.loadTrainingAll()
        return None
    
    def loadTrainingAll(self, shuffleEn = True):
        cnt_file = 0
        for file in self._trainingFilesSet:
            video = vpp.videoProcess(file[1],self._frmSize,cropEn=True,NormEn=True)
            self._trainingVideos = np.append(self._trainingVideos,video,axis=0)
            #labelCode = vpp.int2OneHot(int(file[2]),self._numOfClasses)
            #label = np.repeat(np.reshape(labelCode,(1,self._numOfClasses)),video.shape[0],axis=0)
            label = np.repeat(np.reshape(int(file[2]),(1,1)),video.shape[0],axis=0)
            self._trainingLabels=  np.append(self._trainingLabels,label,axis=0)
            cnt_file+=1
            if cnt_file % 10 == 0:
                print('Loading training videos: ' + str(int(cnt_file * 100 / self._trainingFilesSet.shape[0])) +'%')
        if shuffleEn == True:
            perm = np.arange(self._trainingVideos.shape[0])
            np.random.shuffle(perm)
            self._trainingVideos = self._trainingVideos[perm]
            self._trainingLabels = self._trainingLabels[perm]
        self._trainingVideos = self._trainingVideos - np.mean(self._trainingVideos,axis=(0))
        return None 
    
    def getTrainingSet(self):
        return [self._trainingVideos, self._trainingLabels]
    
    def getEpoch(self):
        return self._trainingEpoch
    
    def resetEpoch(self):
        self._trainingEpoch = 0
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
        #testLabels = np.empty((0,self._numOfClasses),dtype=np.float32)        
        testLabels = np.empty((0,1),dtype=np.float32)        
        for file in self._testingFilesSet:
            #labelCode = vpp.int2OneHot(int(file[2]),self._numOfClasses)
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
                #testLabels = np.append(testLabels,np.reshape(labelCode,(1,self._numOfClasses)),axis=0)
                testLabels = np.append(testLabels,np.reshape(int(file[2]),(1,1)),axis=0)
        testVideos = testVideos - np.mean(testVideos,axis=(0,1))
        return (testVideos.transpose(1,0,2,3,4,5),testLabels)    
    
    def getFileList(self):
        return self._files
            
class ut_interaction_atomic:
    def __init__(self,paths,frmSize):
        self._ut_a0 = ut_interaction([paths[0]], frmSize)
        self._ut_a1 = ut_interaction([paths[1]], frmSize)
        fileList0 = self._ut_a0.getFileList()
        fileList1 = self._ut_a1.getFileList()
        assert np.array_equal(fileList0,fileList1), 'Error, input videos from two set is mismatch!'
    
    def splitTrainingTesting(self,n):
        self._ut_a0.splitTrainingTesting(n,loadTrainingEn=False)
        self._ut_a1.splitTrainingTesting(n,loadTrainingEn=False)
        self._trainingEpoch = 0
        self._trainingPointer = 0
        #self.loadTrainingAll()
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
    
    def getEpoch(self):
        return self._trainingEpoch
    
    def resetEpoch(self):
        self._trainingEpoch = 0
        return None
    
class ut_interaction_ga:
    def __init__(self,paths,frmSize):
        self._ut_g  = ut_interaction([paths[0]], frmSize[0])
        self._ut_a0 = ut_interaction([paths[1]], frmSize[1])
        self._ut_a1 = ut_interaction([paths[2]], frmSize[1])
        fileList_g  = self._ut_g.getFileList()
        fileList_a0 = self._ut_a0.getFileList()
        fileList_a1 = self._ut_a1.getFileList()
        assert np.array_equal(fileList_a0,fileList_a1) and np.array_equal(fileList_g, fileList_a0), 'Error, input videos from three sets are mis-match!'
    
    def splitTrainingTesting(self,n):
        self._ut_g.splitTrainingTesting(n,loadTrainingEn=False)
        self._ut_a0.splitTrainingTesting(n,loadTrainingEn=False)
        self._ut_a1.splitTrainingTesting(n,loadTrainingEn=False)
        self._trainingEpoch = 0
        self._trainingPointer = 0
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
    
    def getEpoch():
        return self._trainingEpoch
        

class ut_interaction_set1(ut_interaction):
    def __init__(self,frmSize,numOfClasses = 6):
        path = [common.path.utSet1Path]
        ut_interaction.__init__(self,path,frmSize, numOfClasses)

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
    def __init__(self,frmSize,numOfClasses=6):
        path = [common.path.utSet2Path]
        ut_interaction.__init__(self,path,frmSize,numOfClasses)

class ut_interaction_set2_atomic(ut_interaction_atomic):
    def __init__(self,frmSize):
        paths = [common.path.utSet2_a0_Path,common.path.utSet2_a1_Path]
        ut_interaction_atomic.__init__(self,paths,frmSize)

class ut_interaction_set2_a(ut_interaction):
    def __init__(self,frmSize):
        paths = [common.path.utSet2_a0_Path,common.path.utSet2_a1_Path]
        ut_interaction.__init__(self,paths,frmSize)

def oneVsRest(y,n):
    y_out = []
    for y_ in y:
        if y_ == n:
            y_ovr = [1,0]
        else:
            y_ovr = [0,1]
        y_out.append(y_ovr)
    return np.array(y_out)

def oneHot(y,numOfClasses):
    y_out = []
    for y_ in y:
        y_oh = vpp.int2OneHot(y_[0], numOfClasses)
        y_out.append(y_oh)
    return np.array(y_out)

if __name__ == '__main__':
    numOfClasses = 6
    ut_set = ut_interaction_set1((112,128,3),numOfClasses=numOfClasses)
    for seq in range(1,11):
        print('seq = ',seq)
        ut_set.splitTrainingTesting(seq,loadTrainingEn=False)
        #ut_set.loadTrainingAll()
        for i in range(10):
            print(i)
            vtr = ut_set.loadTrainingBatch(16)
            for v in vtr[0]:
                vpp.videoPlay(v+0.3)
                print(v)
        
        vt = ut_set.loadTesting()
        y = vt[1]
        print(y)
        print(oneHot(y,numOfClasses))
        for vs in vt[0].transpose(1,0,2,3,4,5):
            for v in vs:
                vpp.videoPlay(v+0.3)
        
    



# *******************************************************************
# for detection task
# *******************************************************************
from mmap import mmap, ACCESS_READ
from xlrd import open_workbook
import cv2

def labelToString(label):
    activitys = ['Hand Shaking', 'Hugging', 'Kicking', 'Pointing', 'Punching', 'Pushing']
    return activitys[label]

# read ground truth from excel file
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
                         
def genNegativeSamples0(setNo,seqNo,NoBias):
    videoName = 'D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_set' + str(setNo) + '/seq' + str(seqNo) +'.avi'
    gt = getGroundTruth(setNo, seqNo)
    cap = cv2.VideoCapture(videoName)
    ret,frame = cap.read()
    frmNo = 0
    gt_i = 0
    frmNeg = 0
    videoCnt = 0
    while ret:
        if gt_i < gt.shape[0]:
            gt_line = gt[gt_i]
            if frmNeg == 0:
                video = np.empty((0, gt_line[6] - gt_line[4], gt_line[5] - gt_line[3], 3)) 
            if frmNo - frmNeg + 64 < gt_line[1]:
                frameChop = frame[gt_line[4]:gt_line[6],gt_line[3]:gt_line[5]]
                video = np.append(video,np.reshape(frameChop,(1,)+frameChop.shape),0)
                if frmNeg == 63:
                    frmNeg = 0
                    videoName = common.path.utSet1Path + 'set' + str(setNo) + '/' + str(videoCnt+NoBias) + '_' + str(seqNo) + '_6.avi'
                    print(videoName)
                    vpp.videoPlay(video.astype(np.uint8))
                    vpp.videoSave(video.astype(np.uint8),videoName)
                    videoCnt+= 1
                else:
                    frmNeg += 1
            if frmNo > gt[gt_i][2]:
                gt_i+=1
        ret,frame = cap.read()
        frmNo += 1
    return videoCnt

def genNegativeSamples1(setNo,seqNo,NoBias):
    videoName = 'D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_set' + str(setNo) + '/seq' + str(seqNo) +'.avi'
    gt = getGroundTruth(setNo, seqNo)
    cap = cv2.VideoCapture(videoName)
    ret,frame = cap.read()
    frmNo = 0
    gt_i = 0
    frmNeg = 0
    videoCnt = 0
    while ret:
        if gt_i < gt.shape[0]:
            gt_line = gt[gt_i]
            if frmNeg == 0:
                video = np.empty((0, gt_line[6] - gt_line[4], int((gt_line[5] - gt_line[3])/2), 3)) 
            if frmNo - frmNeg + 64 < gt_line[1]:
                frameChop = frame[gt_line[4]:gt_line[6],gt_line[3]:gt_line[3]+int((gt_line[5] - gt_line[3])/2)]
                video = np.append(video,np.reshape(frameChop,(1,)+frameChop.shape),0)
                if frmNeg == 63:
                    frmNeg = 0
                    videoName = common.path.utSet1Path + 'set' + str(setNo) + '/' + str(videoCnt+NoBias) + '_' + str(seqNo) + '_6.avi'
                    print(videoName)
                    vpp.videoPlay(video.astype(np.uint8))
                    vpp.videoSave(video.astype(np.uint8),videoName)
                    videoCnt+= 1
                else:
                    frmNeg += 1
            if frmNo > gt[gt_i][2]:
                gt_i+=1
        ret,frame = cap.read()
        frmNo += 1
    return videoCnt
    
#def genNegativeSamples1(setNo,seqNo,NoBias):
#    videoName = 'D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_set' + str(setNo) + '/seq' + str(seqNo) +'.avi'
#    cap = cv2.VideoCapture(videoName)
#    ret,frame = cap.read()
#    h,w = frame.shape[0],frame.shape[1]
#    regions = []
#    regions.append((0,0,300,260))
#    regions.append((int(w/2) - 150, 0, int(w/2) + 150, 260))
#    regions.append((w - 300, 0, w, 260))
#    regions.append((0,h-260,300,h))
#    regions.append((int(w/2) - 150, h - 260, int(w/2) + 150, h))
#    regions.append((w - 300, h - 260, w, h))
#    frmNo = 0
#    videos = np.empty((0,260,300,3)) 
#    while ret:
#        if frmNo >= 200 and frmNo < 263:
#            videos = np.append(videos, np.array([frame[y0:y1,x0:x1] for x0,y0,x1,y1 in regions]),0)
#        if frmNo == 248:
#            videos = np.reshape(videos,(48,6,260,300,3)).transpose(1,0,2,3,4).astype(np.uint8)
#            for i,video in enumerate(videos):
#                videoName = 'set' + str(setNo) + '/' + str(i+NoBias) + '_' + str(seqNo) + '_6.avi'
#                vpp.videoSave(video, videoName)
#                for img in video:
#                    if cv2.waitKey(30) == 27:
#                        break
#        ret,frame = cap.read()
#        frmNo += 1
        
def loadVideo(seqNo):
    if seqNo <= 10:
        videoName = common.path.utSet1DetPath + '/seq' + str(seqNo) +'.avi'
    else:
        videoName = common.path.utSet2DetPath + '/seq' + str(seqNo) +'.avi'
        
    video = vpp.videoRead(videoName,grayMode=False)
    return video
    
def genDetectionBBList(videoIn):
    dMax,yMax,xMax = videoIn.shape[0:3]
    d0,d1 = 0,64
    y0,y1 = 0,260
    x0,x1 = 0,300
    detectionBBList = []
    while d1 < dMax:
        while y1 < yMax:
            while x1 < xMax:
                detectionBBList.append([d0,d1,y0,y1,x0,x1])
                x0 += 32
                x1 += 32
            x0,x1 = 0,300
            y0 += 16 
            y1 += 16 
        x0,x1 = 0,300
        y0,y1 = 0,260
        d0 += 32 
        d1 += 32 
    return detectionBBList
    
if __name__ == '__main__':
    for setNo in (1,):
        NoBias = 60
        videoCnt = 0
        for seqNo in range(1+(setNo-1)*10,11+(setNo-1)*10):
            print(setNo,seqNo,NoBias)
            videoCnt = genNegativeSamples0(setNo,seqNo,NoBias)
            NoBias += videoCnt
            #NoBias += videoCnt
            #videoCnt = genNegativeSamples1(setNo,seqNo,NoBias)
    #setNo,seqNo = 1,1
    #videoName = 'D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_set' + str(setNo) + '/seq' + str(seqNo) +'.avi'
    #gt = getGroundTruth(setNo, seqNo)
    #pred_ibbs = procIBB()
    #print(gt)
    #print(pred_ibbs)
    #cv2.namedWindow('video player')    
    #cap = cv2.VideoCapture(videoName)
    #ret,frame = cap.read()
    #frmNo = 0
    #gt_i = 0
    #ibbs_i = 0
    #video = []
    #while(ret):
    #    if gt_i < gt.shape[0] :
    #        if frmNo > gt[gt_i][1] and frmNo < gt[gt_i][2]:
    #            cv2.rectangle(frame, (gt[gt_i][3], gt[gt_i][4]), (gt[gt_i][5], gt[gt_i][6]), (0, 255, 0), thickness=1)
    #            cv2.putText(frame, labelToString(gt[gt_i][0]), (gt[gt_i][3],gt[gt_i][4] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    #        if frmNo > gt[gt_i][2]:
    #            gt_i+=1
    #    
    #    if ibbs_i < pred_ibbs.shape[0] :
    #        if frmNo > pred_ibbs[ibbs_i][1] and frmNo < pred_ibbs[ibbs_i][2]:
    #            cv2.rectangle(frame, (pred_ibbs[ibbs_i][3], pred_ibbs[ibbs_i][4]), (pred_ibbs[ibbs_i][5], pred_ibbs[ibbs_i][6]), (0, 0, 255), thickness=1)
    #            cv2.putText(frame, labelToString(pred_ibbs[ibbs_i][0]), (pred_ibbs[ibbs_i][5] - 100,pred_ibbs[ibbs_i][4] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    #        if frmNo > pred_ibbs[ibbs_i][2]:
    #            ibbs_i+=1
    #    print(frame.shape)
    #    video.append(frame)
    #    ret,frame = cap.read()
    #    frmNo +=1
    #video = np.array(video)
    #print(video.shape)
    #vpp.videoSave(video, 'seq_1.avi')
        
        
