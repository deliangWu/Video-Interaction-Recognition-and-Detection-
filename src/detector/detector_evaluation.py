import numpy as np
from xlrd import open_workbook
import sys
sys.path.insert(1,'../common')
sys.path.insert(1,'../dataset')
import common
import sp_int_det as sid
import cv2
import videoPreProcess as vpp
import ut_interaction as ut
import matplotlib.pyplot as plt

def plotIbb(cub1,cub2,tMax):
    plt.figure(figsize=(10,5),dpi=72)
    plt.subplots_adjust(hspace=0.3,top = 0.95,bottom = 0.1,left=0.09)
    ax1=plt.subplot(121)
    ax2=plt.subplot(122)
    plt.sca(ax1)
    plt.xlim([0,720])
    plt.ylim([0,480])
    plt.axhline(y=cub1[3], xmin=cub1[2]/720, xmax=cub1[4]/720, color='g',ls ='-',linewidth=1)
    plt.axhline(y=cub1[5], xmin=cub1[2]/720, xmax=cub1[4]/720, color='g',ls ='-',linewidth=1)
    plt.axvline(x=cub1[2], ymin=cub1[3]/480, ymax=cub1[5]/480, color='g',ls ='-',linewidth=1)
    plt.axvline(x=cub1[4], ymin=cub1[3]/480, ymax=cub1[5]/480, color='g',ls ='-',linewidth=1)
    plt.axhline(y=cub2[3], xmin=cub2[2]/720, xmax=cub2[4]/720, color='r',ls ='-',linewidth=1)
    plt.axhline(y=cub2[5], xmin=cub2[2]/720, xmax=cub2[4]/720, color='r',ls ='-',linewidth=1)
    plt.axvline(x=cub2[2], ymin=cub2[3]/480, ymax=cub2[5]/480, color='r',ls ='-',linewidth=1)
    plt.axvline(x=cub2[4], ymin=cub2[3]/480, ymax=cub2[5]/480, color='r',ls ='-',linewidth=1)
    #plt.annotate(str(round(accu,2)),xy=(0+i*0.5,accu),color=color[i][0],size=font2)
    plt.xlabel('x',fontsize=9)
    plt.ylabel('y',fontsize=9)
    
    plt.sca(ax2)
    plt.xlim([0,tMax])
    plt.ylim([0,1])
    plt.axhline(y=0.5,   xmin=cub1[0]/tMax, xmax=cub1[1]/tMax, color='g',ls ='-',linewidth=1)
    plt.axhline(y=0.52, xmin=cub2[0]/tMax, xmax=cub2[1]/tMax, color='r',ls ='-',linewidth=1)
    plt.xlabel('frames',fontsize=9)
    plt.show()

def plotTemp(logName):
    tMax = 2200
    det_ibbSets = np.array(readDetLog(logName))
    gt_ibbSets = []
    for seq in range(1,11):
        gt_ibbSet = np.array(getGroundTruth(1,seq))
        gt_ibbSets.append(gt_ibbSet)
    gt_ibbSets = np.array(gt_ibbSets)
    
    plt.figure(figsize=(7,6),dpi=72)
    plt.subplots_adjust(hspace=0.3,top = 0.95,bottom = 0.1,left=0.09)
    plt.xlim([0,tMax])
    plt.ylim([0,11])
    for i in range(10):
        for item in gt_ibbSets[i]:
            plt.axhline(y=i + 1.1, xmin=item[1]/tMax, xmax=item[2]/tMax, color='g',ls ='-',linewidth=1)
            plt.annotate(str(item[0]),xy=((item[1]+item[2])/2,i+1.15),color='g',size=9)
        for item2 in det_ibbSets[i]:
            plt.axhline(y=i + 0.9, xmin=item2[1]/tMax, xmax=item2[2]/tMax, color='r',ls ='-',linewidth=1)
            plt.annotate(str(item2[0]),xy=((item2[1]+item2[2])/2,i+0.60),color='r',size=9)
        plt.axhline(y=i+1, xmin=0, xmax=1, color='0.8',ls ='--',linewidth=0.3)
    #plt.title('Temporal overlaps and class labels between the detections and the ground truth ',fontsize=11)
    plt.xlabel('frames',fontsize=9)
    plt.ylabel('video sequences',fontsize=9)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, tMax, 200))
    ax.set_yticks(np.arange(0, 11, 1))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    fig_name = 'D:/Course/Final_Thesis_Project/project/Video-Interaction-Recognition-and-Detection-/thesis/chapters/chapter05/fig01/plot_temp.pdf'
    plt.savefig(fig_name)
    print('Figure is saved to ',fig_name)
    plt.show()
    
    
def plotSeq(gt_ibbSet,det_ibbSet,tMax):
    plt.figure(figsize=(10,5),dpi=72)
    
    plt.xlim([0,tMax])
    plt.ylim([0,1])
    for item in gt_ibbSet:
        plt.axhline(y=0.52, xmin=item[1]/tMax, xmax=item[2]/tMax, color='g',ls ='-',linewidth=1)
    for item2 in det_ibbSet:
        plt.axhline(y=0.48, xmin=item2[1]/tMax, xmax=item2[2]/tMax, color='r',ls ='-',linewidth=1)
    plt.xlabel('frames',fontsize=9)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, tMax, tMax//8))
    plt.xticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.show()

# read ground truth from excel file
def getGroundTruth(setNo, seqNo):
    workbook = open_workbook(common.path.projectPath + 'datasets/UT_Interaction/ut-interaction_labels_110912.xls')
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

def cubOverlapRation(cub1,cub2):
    volume1 = (cub1[1] - cub1[0]) * (cub1[4] - cub1[2]) * (cub1[5] - cub1[3])
    volume2 = (cub2[1] - cub2[0]) * (cub2[4] - cub2[2]) * (cub2[5] - cub2[3])    
    l = max(0,min(cub1[1],cub2[1]) - max(cub1[0],cub2[0]))
    
    xA = max(cub1[2],cub2[2])
    yA = max(cub1[3],cub2[3])
    xB = min(cub1[4],cub2[4])
    yB = min(cub1[5],cub2[5])
    if xB > xA and yB > yA:
        area = (xB - xA) * (yB - yA)
    else:
        area = 0
    volume = area * l
    overlapRatio = volume / (volume1 + volume2 - volume)
    l_ratio = l/(cub1[1] - cub1[0] + cub2[1] - cub2[0] - l)
    a_ratio = area / ((cub1[4] - cub1[2]) * (cub1[5] - cub1[3]) + (cub2[4] - cub2[2]) * (cub2[5] - cub2[3]) - area)
    return overlapRatio

def rectOverlapRation(rect1,rect2):
    area1 =  (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 =  (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])    
    xA = max(rect1[0],rect2[0])
    yA = max(rect1[1],rect2[1])
    xB = min(rect1[2],rect2[2])
    yB = min(rect1[3],rect2[3])
    if xB > xA and yB > yA:
        area = (xB - xA) * (yB - yA)
    else:
        area = 0
    overlapRatio = area / (area1 + area2 - area)
    return overlapRatio

def readDetLog(fname):
    with open(fname) as f:
        content = f.readlines()
    st_ind = False
    ibbSets = []
    for i in range(10):
        ibbSets.append([])
    for line in content:
        if 'sequence is' in line:
            seq = int(line[line.index('is') + 3:]) - 1
            loadEn = True
        elif '[' in line and loadEn:
            ibbSet = [int(item.rstrip(']')) for item in line.split()[1:8]]
            ibbSets[seq].append(ibbSet)
        elif 'prob' in line:
            loadEn = False
    ibbSets = np.array(ibbSets)
    return ibbSets

'''calculate the precision and recall for the given ground truth and detections'''
def calPR(gt_ibbSet, det_ibbSet):
    correctNo = 0
    for item in det_ibbSet:
        det_label = item[0]
        det_cub = item[1:]
        if det_label in list(gt_ibbSet[:,0]):
            gt_index = list(gt_ibbSet[:,0]).index(det_label)
            gt_cub = gt_ibbSet[gt_index][1:]
            overlapRatio = cubOverlapRation(det_cub, gt_cub)
            if overlapRatio >= 0.5:
                correctNo += 1
    precision = correctNo / det_ibbSet.shape[0]
    recall = correctNo / gt_ibbSet.shape[0]
    print('precison = ', precision, ' and recall = ', recall)
    return (precision,recall)
        
        
if __name__ == '__main__#':
    logName = common.path.logPath + 'c3d_detector_06-15-21-21.txt'
    prList = []
    det_ibbSets = np.array(readDetLog(logName))
    for seq in range(1,11):
        det_ibbSet = np.array(det_ibbSets[seq-1])
        if det_ibbSet != []:
            print('******************** seq ' + str(seq) + '*******************')
            gt_ibbSet = np.array(getGroundTruth(1,seq))
            precision,recall = calPR(gt_ibbSet, det_ibbSet)
            prList.append([precision,recall])
    prList = np.array(prList)
    
    print('The overall precision is ', np.mean(prList[:,0]))
    print('The overall recall is ', np.mean(prList[:,1]))
    plotTemp(logName)
    
if __name__ == '__main__s':
    for seq in range(1,11):
        spatialMeasure(seq)
        
if __name__ == '__main__':
    for seq in range(1,2):
        genIDet(1,'seq_idet_'+str(seq)+'.avi')
    
    