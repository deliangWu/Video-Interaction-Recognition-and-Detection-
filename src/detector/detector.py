from __future__ import print_function
#from __future__ import division
import numpy as np
import os
import sys
sys.path.insert(1,'/home/wdl/opencv/lib')
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import tensorflow as tf
import sys
from os.path import join 
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
sys.path.insert(1,'../common')
sys.path.insert(1,'../train')
sys.path.insert(1,'../humanSegmentation')
import ut_interaction as ut
import model
import common
import network
import videoPreProcess as vpp
import humanDetectionAndTracking as hdt
import sp_int_det as sid
import time

def main(argv):
    # ***********************************************************
    # define the network
    # ***********************************************************
    numOfClasses = 7 
    frmSize = (112,128,3)
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize,nof_conv1=32, nof_conv2= 128, nof_conv3=256, nof_conv4= 512, noo_fc6=4096, noo_fc7=4096)
    # ***********************************************************
    # define session
    # ***********************************************************
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    initVars = tf.global_variables_initializer()
   
    # ***********************************************************
    # detector
    # ***********************************************************
    savePrefix = 'c3d_detector_'
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    if len(argv) >= 2 and argv[1][0:3] == 'seq':
        seqNo = int(argv[1][3:])
        seqRange = (seqNo,)
    else:
        seqRange = range(1,11)
    for seq in seqRange:
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        with sess.as_default():
            sess.run(initVars)
        # load trained network  
        saver_net = tf.train.Saver()
        saver_net.restore(sess, join(common.path.variablePath, 'c3d_c7_det_set1_1_det7c.ckpt'))
        # get bounding boxes for the interacting people        
        boundingBoxes,bbInitFrmNo,video = sid.spIntDet(seq)
        for vLen in(64,):
            for stride in (8,):
                common.pAndWf(logName, '*********** vLen :'+str(vLen)+' stride: '+ str(stride) +' **************************\n')
                ibbLists= sid.genIBB(boundingBoxes,vLen,stride)
                #dispIBB(video, bbInitFrmNo, ibbLists[:,2])
                # generate the predicted labels for candidate bounding boxes
                pred_yList = sid.pred_IBB(video,ibbLists[:,2], bbInitFrmNo,sess,c3d,vLen,stride)
                # combine the temporal-neighbour bounding boxes as a same interaction label
                ibbSets = sid.comb_IBB(pred_yList,vLen)
                # non-maximum suppression to vote the most possible lables
                finalPredIBB,probs = sid.NMS_IBB(ibbSets)
                f2p = sid.pred_IBB2(video, finalPredIBB, sess, c3d)
                common.pAndWf(logName,str(f2p)+'\n' + 'prob matrix is \n' + str(probs) + '\n')
 
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)