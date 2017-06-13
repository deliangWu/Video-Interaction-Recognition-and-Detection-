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
sys.path.insert(1,'../humanSegmentation')
import ut_interaction as ut
import model
import common
import network
import videoPreProcess as vpp
import humanDetectionAndTracking as hdt
import time

def readDetLog(fname):
    with open(fname) as f:
        content = f.readlines()
    st_ind = False
    for line in content:
        if 'vLen' in line:
            if st_ind is False:
                ibbSets= []
                st_ind = True
            else:
                break
        elif st_ind :
            ibbSet = [int(item.rstrip(']')) for item in line.split()[1:8]]
            ibbSets.append(ibbSet)
    ibbSets = np.array(ibbSets)
    return ibbSets

def pred_IBB2(video,ibbSets,sess,c3d): 
    pred_ibb2List = []
    for ibbSet in ibbSets:
        ibb = ibbSet[3:7]
        vChop = video[ibbSet[1]:ibbSet[2],ibb[1]:ibb[3],ibb[0]:ibb[2]]
        vChop = vpp.videoProcessVin(vChop, (112,128,3), downSample=0, RLFlipEn=False,numOfRandomCrop=4)
        vChop = vpp.videoNorm1(vChop,normMode=1)
        vChop_det = np.reshape(vChop,(-1,1,16,112,128,3))
        prob = c3d.evaluateProb(vChop_det, sess)[0]
        pred_y = np.argmax(prob)
        pred_ibb2List.append([pred_y, ibbSet[1],ibbSet[2],ibb[0],ibb[1],ibb[2],ibb[3]])
    return pred_ibb2List

def main(argv):
    # ***********************************************************
    # define the network
    # ***********************************************************
    numOfClasses = 6 
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
    # define the dataset
    # ***********************************************************
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    savePrefix = 'c3d_detector2_'
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    seqNo = int(argv[1][3:])
    seqRange = (seqNo,)
    #seqRange = (1,)
    for seq in seqRange:
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        with sess.as_default():
            sess.run(initVars)
        # load trained network  
        saver_net = tf.train.Saver()
        saver_net.restore(sess, join(common.path.variablePath, 'c3d_train_on_ut_set1_' + str(seq) + '.ckpt'))
        
        # generate candidate bounding boxe by applying person detection and tracking            
        video = ut.loadVideo(seq)
        logName = common.path.logPath + 'c3d_detector_06-13-21-15.txt'
        ibbSets = readDetLog(logName)
        print('the original ibbSets is ', ibbSets)
        finalIbbSets = pred_IBB2(video, ibbSets, sess, c3d)
        print('the new ibbSets is ', finalIbbSets)
 
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)