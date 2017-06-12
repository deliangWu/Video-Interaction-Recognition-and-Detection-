from __future__ import print_function
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import sys
sys.path.insert(1,'../common')
sys.path.insert(1,'../datasets')
import common
import videoPreProcess as vpp
import cv2

def transVec(vecIn):
    vec_exp = np.exp(-vecIn)
    vecOut = np.divide((1-vec_exp),(1+vec_exp))
    return vecOut

def readLog(logName,seqBias = 1, start = 0, end = 1):
    def alignList(listIn):
        listIn_resize = list(listIn)
        for accus in listIn_resize:
            accus.extend([accus[-1]]*(maxLen-len(accus)))
        listIn_resize = np.array(listIn_resize)
        return listIn_resize
    
    with open(logName) as f:
        content = f.readlines()
    stepList = []
    t_accuList = []
    tr_accuList = []
    t_lossList = []
    tr_lossList = []
    seq = 0
    loading = False
    for line in content:
        line = line.strip()
        if 'RUN '+str(end) in line:
            break
        if 'RUN '+str(start) in line or loading is True:
            loading = True
            if 'current sequence' in line:
                seq = int(line[line.index('is') + 3:]) - seqBias
                stepList.append([])
                t_accuList.append([])
                tr_accuList.append([])
                t_lossList.append([])
                tr_lossList.append([])
            elif 'step' in line:
                step = int(line[line.index('step') + 6:line.index('step:') + line[line.index('step:'):].index(',')])
                stepList[seq].append(step)
                t_accu = float(line[line.index('testing:') + 9:line.index('testing:') + line[line.index('testing:'):].index(',')])
                t_accuList[seq].append(t_accu)
                tr_accu = float(line[line.index('training:') + 10:line.index('training:') + line[line.index('training:'):].index(',')])
                tr_accuList[seq].append(tr_accu)
                t_loss = float(line[line.index('loss_t:') + 8:line.index('loss_t:') + line[line.index('loss_t:'):].index(',')])
                t_lossList[seq].append(t_loss)
                tr_loss = float(line[line.index('loss_tr:') + 9:line.index('loss_tr:') + line[line.index('loss_tr:'):].index(',')])
                tr_lossList[seq].append(tr_loss)
            
    
    lenList = [len(steps) for steps in stepList]
    maxLen = max(lenList)
    steps = stepList[lenList.index(maxLen)]
    steps = np.array(steps) * 16 / 1296
   
    print(t_accuList) 
    t_accuList_resize = alignList(t_accuList)
    print(t_accuList_resize.shape)
    t_accus = np.mean(t_accuList_resize,0)
    
    tr_accuList_resize = alignList(tr_accuList)
    tr_accus = np.mean(tr_accuList_resize,0)
    
    
    t_lossList_resize = alignList(t_lossList)
    t_loss = np.array([t_loss_i/np.max(t_loss_i) for t_loss_i in t_lossList_resize])
    t_loss = np.mean(t_loss,0)
    
    tr_lossList_resize = alignList(tr_lossList)
    tr_loss = np.array([tr_loss_i/np.max(tr_loss_i) for tr_loss_i in tr_lossList_resize])
    tr_loss = np.mean(tr_loss,0)
    #assert accuList_resize.shape[0] == 6
    return (steps,t_accus,tr_accus,t_loss,tr_loss)

    
def plot1(figPlot,dispEn=True):
    # common definitions
    label = []
    log = []
    color=['r-','g-','b-','c-','y-']
    plt.figure(figsize=(6.6,5),dpi=72)
    plt.subplots_adjust(hspace=0.3,top = 0.95,bottom = 0.1,left=0.09)
    ax1=plt.subplot(221)
    ax2=plt.subplot(222)
    ax3=plt.subplot(223)
    ax4=plt.subplot(224)
    font1,font2 = 9,7
    lw = 0.7
    
    if figPlot == 'biases':
        # initial biaes 
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-13-25.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-14-08.txt'          # base
        label.append('initial biases = 0.1')
        label.append('initial biases = 0.01')
        label.append('initial biases = 0.001')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=0,end=1))
        log.append(readLog(fname_2,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_biases.pdf'
    
    if figPlot == 'lr':
        # learning rate 
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-10-28.txt'
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-08-16.txt'          # base
        fname_3 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-12-13.txt'
        label.append('lr=1e-3')
        label.append('lr=1e-4')
        label.append('lr=1e-4 with decay')
        label.append('lr=1e-5')
        log.append(readLog(fname_0,start=0,end=1))
        log.append(readLog(fname_1,start=1,end=2))
        log.append(readLog(fname_2,start=1,end=2))
        log.append(readLog(fname_3,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_lr.pdf'
    
    if figPlot == 'dropout':
        # dropout 
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-15-03.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-16-13.txt'
        label.append('keep prob = 0.5')
        label.append('keep prob = 0.8')
        label.append('keep prob = 0.2')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=1,end=2))
        log.append(readLog(fname_2,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_dropout.pdf'
    
    if figPlot == 'layer': 
        # cnn layers 
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-16-30.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-17-07.txt'
        label.append('4 3DConv layers')
        label.append('5 3DConv layers')
        label.append('6 3DConv layers')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=1,end=2))
        log.append(readLog(fname_2,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_layers.pdf'
    
    if figPlot == 'kernel': 
        # cnn kernel size
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-17-54.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-17-57.txt'
        fname_3 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-19-46.txt'
        label.append('3x 3x3 convolutional kernels')
        label.append('4x 4x3 convolutional kernels')
        label.append('4x 4x4 convolutional kernels')
        label.append('3x 4x4 convolutional kernels')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=1,end=2))
        log.append(readLog(fname_2,start=1,end=2))
        log.append(readLog(fname_3,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_cnn_kernel.pdf'
    
    if figPlot == 'nof':
        # number of filters 
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-19-53.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-22-38.txt'
        fname_3 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-01-43.txt'
        fname_4 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-01-45.txt'
        label.append('[32,128,256,512]')
        label.append('[64,128,256,512]')
        label.append('[32,64,256,512]')
        label.append('[32,128,512,512]')
        label.append('[32,128,256,256]')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=1,end=2))
        log.append(readLog(fname_2,start=2,end=3))
        log.append(readLog(fname_3,start=1,end=2))
        log.append(readLog(fname_4,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_nof.pdf'
    
    if figPlot == 'noo':
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-09-05.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-09-07.txt'
        label.append('[4096,4096]')
        label.append('[4096,2048]')
        label.append('[2048,2048]')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=0,end=1))
        log.append(readLog(fname_2,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_noo.pdf'
    
    if figPlot == 'RLFlipping':
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-11-27.txt'
        label.append('With RLFlipping')
        label.append('Without RLFlipping')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=0,end=1))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_rlf.pdf'
    
    if figPlot == 'RandomCropping':
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-15-39.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-11-30.txt'
        label.append('random cropping = 4')
        label.append('random cropping = 2')
        label.append('random cropping = 1')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=0,end=1))
        log.append(readLog(fname_2,start=3,end=4))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_rc.pdf'
    
    if figPlot == 'downSampling':
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-15-50.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-17-10.txt'
        fname_3 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-17-12.txt'
        label.append('temporal dowmSampling = 0')
        label.append('temporal downSampling = 1')
        label.append('temporal downSampling = 2')
        label.append('temporal downSampling = 3')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=1,end=2))
        log.append(readLog(fname_2,start=0,end=1))
        log.append(readLog(fname_3,start=1,end=2))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_tds.pdf'
    
    if figPlot == 'dataNormMode':
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set1_06-10-11-02.txt'          # base
        fname_1 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-18-43.txt'
        fname_2 = common.path.logPath + 'c3d_train_on_ut_set1_06-11-18-47.txt'
        label.append('X - mean')
        label.append('(X - mean)/std')
        label.append('X')
        log.append(readLog(fname_0,start=1,end=2))
        log.append(readLog(fname_1,start=4,end=5))
        log.append(readLog(fname_2,start=4,end=5))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_dnm.pdf'
    
    if figPlot == 'set2':
        fname_0 = common.path.logPath + 'c3d_train_on_ut_set2_06-12-08-50.txt'          # base
        label.append('set2')
        log.append(readLog(fname_0,seqBias=11,start=1,end=2))
        fig_name = 'D:/Course/Final_Thesis_Project/Thesis/chapters/chapter05/fig01/plot_set2.pdf'
    
    # ************************************************************* 
    # plot evaluating accuracy
    # ************************************************************* 
    plt.sca(ax1)
    for i,logi in enumerate(log):
        accu = np.mean(logi[1][-16:])
        plt.axhline(y=accu, xmin=0, xmax=7, hold=None,color=color[i][0],ls ='--',linewidth=lw)
        #plt.annotate(str(round(accu,2)),xy=(0+i*0.5,accu),color=color[i][0],size=font2)
        plt.plot(logi[0],logi[1], color[i], label = label[i],linewidth=lw)
        print('Accuracy = ' ,np.mean(logi[1][-20:]))
    plt.title('Evaluating accuray',fontsize=font1)
    plt.xlabel('Training epochs',fontsize=font2)
    plt.ylabel('Classification accuracy',fontsize=font2)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 7.1, 1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    plt.xticks(fontsize=font2)
    plt.yticks(fontsize=font2)
    plt.legend(fontsize=font2)
    plt.grid(b=1)
    # ************************************************************* 
    # plot evaluating loss 
    # ************************************************************* 
    plt.sca(ax2)  
    for i,logi in enumerate(log):
        plt.plot(logi[0],logi[3], color[i], label = label[i],linewidth=lw)
    plt.title('Evaluating loss',fontsize=font1)
    plt.xlabel('Training epochs',fontsize=font2)
    plt.ylabel('Loss',fontsize=font2)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 7.1, 1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    plt.xticks(fontsize=font2)
    plt.yticks(fontsize=font2)
    plt.legend(fontsize=font2)
    plt.grid(b=6)
    
    # ************************************************************* 
    # plot training accuracy
    # ************************************************************* 
    plt.sca(ax3)
    for i,logi in enumerate(log):
        plt.plot(logi[0],logi[2], color[i], label = label[i],linewidth=lw)
    plt.title('Training accuracy',fontsize=font1)
    plt.xlabel('Training epochs',fontsize=font2)
    plt.ylabel('Classification accuracy',fontsize=font2)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 7.1, 1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    plt.xticks(fontsize=font2)
    plt.yticks(fontsize=font2)
    plt.legend(fontsize=font2)
    plt.grid(b=6)
    # ************************************************************* 
    # plot training loss 
    # ************************************************************* 
    plt.sca(ax4)
    for i,logi in enumerate(log):
        plt.plot(logi[0],logi[4], color[i], label = label[i],linewidth=lw)
    plt.title('Training loss',fontsize=font1)
    plt.xlabel('Training epochs',fontsize=font2)
    plt.ylabel('Loss',fontsize=font2)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 7.1, 1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    plt.xticks(fontsize=font2)
    plt.yticks(fontsize=font2)
    plt.legend(fontsize=font2)
    plt.grid(b=6)
    
    # ***********************************************************
    # ***********************************************************
    plt.savefig(fig_name)
    print('Figure is saved to ',fig_name)
    if dispEn:
        plt.show()
        
    #figPlot = 'biases'
    #figPlot = 'lr'
    #figPlot = 'dropout'
    #figPlot = 'layer'
    #figPlot = 'kernel' 
    #figPlot = 'nof'
    #figPlot = 'noo'
    #figPlot = 'RLFlipping'
    #figPlot = 'RandomCropping'
    #figPlot = 'downSampling'
    #figPlot = 'dataNormMode'
figPlotList = ['biases','lr','dropout','layer','kernel', 'nof','noo','RLFlipping','RandomCropping','downSampling','dataNormMode']
plot1('set2',dispEn=True)

def dispPersonDetector():
    fileName = 'D:/Course/Final_Thesis_Project/project/Video-Interaction-Recognition-and-Detection-/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/vOut/1_1_2.avi'
    fileName = 'D:/Course/Final_Thesis_Project/project/Video-Interaction-Recognition-and-Detection-/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/vOut/41_8_0.avi'
    fileName = 'D:/Course/Final_Thesis_Project/project/Video-Interaction-Recognition-and-Detection-/src/train/seq_10.avi'
    video = vpp.videoRead(fileName,grayMode=False)
    print(video.shape)
    sample = np.arange(200,video.shape[0]-400,int((video.shape[0]-600)/8))
    video = video[sample]
    resizeRatio_w = 0.2
    resizeRatio_h = 0.2
    img = video[0]
    img = cv2.resize(img,(int(img.shape[1] * resizeRatio_w), int(img.shape[0] * resizeRatio_h)), interpolation= cv2.INTER_AREA)
    frames = img 
    for img in video[1:]:
        img = cv2.resize(img,(int(img.shape[1] * resizeRatio_w), int(img.shape[0] * resizeRatio_h)), interpolation= cv2.INTER_AREA)
        frames = np.append(frames,img,1) 
    cv2.namedWindow('img',1)
    while True:
        cv2.imshow('img',frames)
        cv2.waitKey(30)
    
#dispPersonDetector()   