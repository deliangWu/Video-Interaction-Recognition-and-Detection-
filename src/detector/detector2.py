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
sys.path.insert(1,'../train')
import ut_interaction as ut
import model
import common
import network
import videoPreProcess as vpp
import humanDetectionAndTracking as hdt
import time
from collections import Counter
import detector_evaluation as det_eva


def pred_IBB2(video,ibbSets,sess,c3d): 
    pred_ibb2List = []
    for ibbSet in ibbSets:
        ibb = ibbSet[3:7]
        yList = []
        for i in range(5):
            vChop = video[ibbSet[1]:ibbSet[2],ibb[1]:ibb[3],ibb[0]:ibb[2]]
            vChop = vpp.videoProcessVin(vChop, (112,128,3), downSample=0, RLFlipEn=False,numOfRandomCrop=4)
            vChop = vpp.videoNorm1(vChop,normMode=1)
            vChop_det = np.reshape(vChop,(-1,1,16,112,128,3))
            prob = c3d.evaluateProb(vChop_det, sess)[0]
            pred_y = np.argmax(prob)
            yList.append(pred_y)
        pred_label = Counter(yList).most_common(1)[0][0]
        
        pred_ibb2List.append([pred_label, ibbSet[1],ibbSet[2],ibb[0],ibb[1],ibb[2],ibb[3]])
    return np.array(pred_ibb2List)

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
   
    savePrefix = 'c3d_detector2_'
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    # load the results of 2-classes (interacting or not) interaction detection
    d2cLogName = common.path.logPath + argv[2] 
    ibbSets = det_eva.readDetLog(d2cLogName)
    if len(argv) >= 2 and argv[1][0:3] == 'seq':
        seqNo = int(argv[1][3:])
        seqRange = (seqNo,)
    else:
        seqRange = range(1,11)
        
    # ***************************************************    
    # Run detector    
    # ***************************************************    
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
        # load the unsegmented video 
        video = ut.loadVideo(seq)
        finalIbbSets = pred_IBB2(video, ibbSets[seq-1], sess, c3d)
        common.pAndWf(logName,str(finalIbbSets)+'\n')
 
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)