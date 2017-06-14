import numpy as np
from xlrd import open_workbook
import sys
sys.path.insert(1,'../common')
import common

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

def plotTemp(gt_ibbSets,det_ibbSets,tMax):
    plt.figure(figsize=(10,5),dpi=72)
    
    plt.xlim([0,tMax])
    plt.ylim([0,11])
    for i in range(10):
        for item in gt_ibbSets[i]:
            plt.axhline(y=i + 1.1, xmin=item[1]/tMax, xmax=item[2]/tMax, color='g',ls ='-',linewidth=1)
        for item2 in det_ibbSets[i]:
            plt.axhline(y=i + 0.9, xmin=item2[1]/tMax, xmax=item2[2]/tMax, color='r',ls ='-',linewidth=1)
    plt.xlabel('frames',fontsize=9)
    plt.ylabel('sequences',fontsize=9)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, tMax, tMax//8))
    ax.set_yticks(np.arange(0, 11, 1))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
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
    l = min(cub1[1],cub2[1]) - max(cub1[0],cub2[0])
    
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
    print('ratio of l is ', l_ratio)
    print('ratio of area is ', a_ratio)
    print('l * a ', l_ratio * a_ratio)
    
    print(overlapRatio)

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
        elif '[' in line:
            ibbSet = [int(item.rstrip(']')) for item in line.split()[1:8]]
            ibbSets[seq].append(ibbSet)
    ibbSets = np.array(ibbSets)
    return ibbSets
    
#ibbList= [[  4,  735,  867,  222,  126 , 639 , 425], \
#     [   2,  899, 1055,  227,  122 , 621 , 421],  \
#     [   1, 1039, 1227,  296,  124 , 630 , 423],  \
#     [   0, 1395, 1555,  232,  112 , 625 , 411],  \
#     [   5, 1555, 1695,  233,  115 , 664 , 414]]    
#
#
#for ibb in ibbList[0:]:
#    print('*********************************************')
#    gt_ibb = gt[list(gt[:,0]).index(ibb[0])]
#    cub1 = gt_ibb[1:]
#    cub2 = ibb[1:]
#    tMax = max(np.max(gt[:,2]),np.max(np.array(ibbList)[:,2]))
#    cubOverlapRation(cub1, cub2)
#    plotIbb(cub1,cub2,tMax)
logName = common.path.logPath + 'c3d_detector_06-14-13-27.txt'
det_ibbSets = readDetLog(logName)
gt_ibbSets = []
for seq in range(1,11):
    gt_ibbSets.append(getGroundTruth(1,seq))
plotTemp(gt_ibbSets, det_ibbSets, 2200)