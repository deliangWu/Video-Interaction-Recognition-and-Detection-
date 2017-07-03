'''generates some videos for presentation'''
import numpy as np
from xlrd import open_workbook
import sys
sys.path.insert(1,'../common')
sys.path.insert(1,'../dataset')
sys.path.insert(1,'../humanSegmentation')
import common
import sp_int_det as sid
import cv2
import videoPreProcess as vpp
import ut_interaction as ut
import matplotlib.pyplot as plt
import ut_interaction as ut
import model
import common
import network
import videoPreProcess as vpp
import humanDetectionAndTracking as hdt
import detector_evaluation as det_ev
import time
from collections import Counter


''' draw detected bounding boxes of each individual '''
def genPDet(seq,fname):
    # generate candidate bounding boxe by applying person detection and tracking            
    video = ut.loadVideo(seq)
    picks = common.readListFromFile('./bbList/bbList_seq'+str(seq)+'.txt')
    for img,pick in zip(video,picks):
        for bb in pick:
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(0,0,255),1)
    vpp.videoSave(video,fname)
    
'''generate video with bounding boxes after interacting people detecion'''
def genIPDet(seq,fname):
    video = ut.loadVideo(seq)
    picks = common.readListFromFile('./bbList/bbList_seq'+str(seq)+'.txt')
    picks = sid.normBB(picks,video)
    vpp.videoSave(video,fname)

'''generate video with bounding boxes after tracking'''
def genIPFDet(seq,fname):
    video = ut.loadVideo(seq)
    picks = common.readListFromFile('./bbList/bbList_seq'+str(seq)+'.txt')
    picks = sid.normBB(picks)
    _,_,_,boundingBoxes,bbInitFrmNo = hdt.humanTracking(video,picks,dispBBEn = True) 
    vpp.videoSave(video,fname)

'''generate video with bounding boxes after tracking'''
def spatialMeasure(seq):
    boundingBoxes,bbInitFrmNo,video = sid.spIntDet(seq)
    cv2.namedWindow('disp')
    gt_i = 0
    gt = getGroundTruth(1, seq)
    ratioList = []
    for i,img in enumerate(video):
        if i > bbInitFrmNo:
            # generate ibb according to the bbs
            bb0 = boundingBoxes[i-bbInitFrmNo][0][0]
            bb1 = boundingBoxes[i-bbInitFrmNo][1][0]
            bb0_mean = np.array(bb0).astype(np.uint16)
            bb1_mean = np.array(bb1).astype(np.uint16)
            x_center_ibb = int(np.mean([bb0_mean[0],bb1_mean[2]]))
            y_center_ibb = int(np.mean([bb0_mean[1],bb1_mean[1],bb0_mean[3],bb1_mean[3]]))
            h_ibb = int((np.mean([bb0_mean[3],bb1_mean[3]]) - np.mean([bb0_mean[1],bb1_mean[1]])) * 1.0)
            w_ibb_min = int(h_ibb * 128/112 * 1.0)
            w_ibb_max = int(h_ibb * 128/112 * 1.3)
            w_ibb = min(max(w_ibb_min,int(bb1_mean[2] - bb0_mean[0])),w_ibb_max)
            ibb = [max(0,x_center_ibb-int(w_ibb/2)),max(0,y_center_ibb-int(h_ibb/2)),min(720,x_center_ibb+int(w_ibb/2)),min(480,y_center_ibb+int(h_ibb/2))]
            # draw ibb on the image
            cv2.rectangle(img, (ibb[0],ibb[1]),(ibb[2],ibb[3]), (0, 0, 255), thickness=4)
            
            # draw gt on the image
            if gt_i < gt.shape[0] :
                if i > gt[gt_i][1] and i < gt[gt_i][2]:
                    cv2.rectangle(img, (gt[gt_i][3], gt[gt_i][4]), (gt[gt_i][5], gt[gt_i][6]), (0, 255, 0), thickness=4)
                    #cv2.putText(img, labelToString(gt[gt_i][0]), (gt[gt_i][3],gt[gt_i][4] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    ratio = rectOverlapRation(ibb, [gt[gt_i][3],gt[gt_i][4],gt[gt_i][5],gt[gt_i][6]])
                    ratioList.append(ratio)
                if i > gt[gt_i][2]:
                    gt_i+=1
    meanRatio = np.mean(np.array(ratioList))
    print('Seq', seq, ', the overall average overlap of all frames is ',meanRatio)
    vpp.videoSave(video,'seq_ibb_B_'+str(seq)+'.avi')


'''generate video with bounding boxes of the detections and the ground truth'''
def genIDet(seq,fname):
    gt_i = 0
    ibbs_i = 0
    gt = det_ev.getGroundTruth(1, seq)
    logName = common.path.logPath + 'c3d_detector_06-15-21-21.txt'
    pred_ibbs = np.array(det_ev.readDetLog(logName)[seq-1])
    video = ut.loadVideo(seq)
    for i,img in enumerate(video):
        # draw gt on the image
        if gt_i < gt.shape[0] :
            if i > gt[gt_i][1] and i < gt[gt_i][2]:
                cv2.rectangle(img, (gt[gt_i][3], gt[gt_i][4]), (gt[gt_i][5], gt[gt_i][6]), (0, 255, 0), thickness=1)
                cv2.putText(img, ut.labelToString(gt[gt_i][0]), (gt[gt_i][3],gt[gt_i][4] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,0), 1, cv2.LINE_AA)
            if i > gt[gt_i][2]:
                gt_i+=1
    
        if ibbs_i < pred_ibbs.shape[0] :
            if i > pred_ibbs[ibbs_i][1] and i < pred_ibbs[ibbs_i][2]:
                cv2.rectangle(img, (pred_ibbs[ibbs_i][3], pred_ibbs[ibbs_i][4]), (pred_ibbs[ibbs_i][5], pred_ibbs[ibbs_i][6]), (0, 0, 255), thickness=1)
                cv2.putText(img, ut.labelToString(pred_ibbs[ibbs_i][0]), (pred_ibbs[ibbs_i][5] - 100,pred_ibbs[ibbs_i][4] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 1, cv2.LINE_AA)
            if i > pred_ibbs[ibbs_i][2]:
                ibbs_i+=1
    vpp.videoSave(video,fname)

'''Combine four videos into one'''
def videoCombine(v1,v2,v3,v4,fname):
    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)
    cap3 = cv2.VideoCapture(v3)
    cap4 = cv2.VideoCapture(v4)
    ret1,f1 = cap1.read()
    ret2,f2 = cap1.read()
    ret3,f3 = cap1.read()
    ret4,f4 = cap1.read()
    
    frmSize = (720*2,480*2)
    if cv2.__version__  == '3.2.0':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fname, fourcc, 20.0,frmSize)
    else:
        out = cv2.VideoWriter(fname, cv2.cv.CV_FOURCC('X','V','I','D'),20,frmSize)
        
    while ret1:
        f12 = np.concatenate([f1,f2],1)
        f34 = np.concatenate([f3,f4],1)
        fo = np.concatenate([f12,f34],axis=0)
        fo_4d = np.reshape(fo,(1,)+fo.shape)
        cv2.imshow('img',fo)
        cv2.waitKey(1)
        ret1,f1 = cap1.read()
        ret2,f2 = cap2.read()
        ret3,f3 = cap3.read()
        ret4,f4 = cap4.read()
        out.write(fo)
    out.release()
    cv2.destroyAllWindows()
        
for seq in range(9,11):
    print('******************* seq ', seq,'********************')
    genPDet(seq, './genV/seq_pdet_'+str(seq)+'.avi')
    print('./genV/seq_pdet_'+str(seq)+'.avi')
    genIPDet(seq, './genV/seq_ipdet_'+str(seq)+'.avi')
    print('./genV/seq_ipdet_'+str(seq)+'.avi')
    genIPFDet(seq, './genV/seq_ipfdet_'+str(seq)+'.avi')    
    print('./genV/seq_ipfdet_'+str(seq)+'.avi')
    genIDet(seq, './genV/seq_idet_'+str(seq)+'.avi')    
    print('./genV/seq_idet_'+str(seq)+'.avi')
    v1 = './genV/seq_pdet_'+str(seq)+'.avi'
    v2 = './genV/seq_ipdet_'+str(seq)+'.avi'
    v3 = './genV/seq_ipfdet_'+str(seq)+'.avi'
    v4 = './genV/seq_idet_'+str(seq)+'.avi'
    print('start to combine videos')
    videoCombine(v1, v2, v3, v4,'./genV/seq_o_'+str(seq)+'.avi')