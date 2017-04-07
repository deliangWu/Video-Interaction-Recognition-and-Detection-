
from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import sys
from os.path import join 
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
sys.path.insert(1,'../common')
import ut_interaction as ut
import model
import common
import network
import videoPreProcess as vpp
import time

def main(argv):
    # ***********************************************************
    # define the network
    # ***********************************************************
    numOfClasses = 7 
    frmSize = (112,128,3)
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize)
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
    savePrefix = 'c3d_detector_'
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    seqRange = range(1,2)
    
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        video = ut.loadVideo(seq)
        detBBList = ut.genDetectionBBList(video)

        saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
        saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
        saver_feature_g.restore(sess,join(common.path.variablePath, 'c3d_train_on_ut_set1_' + str(seq) + '_fg.ckpt'))
        saver_classifier.restore(sess,join(common.path.variablePath, 'c3d_train_on_ut_set1_' + str(seq) + '_c7.ckpt'))
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        pred_yList = []
        for d0,d1,y0,y1,x0,x1 in detBBList:
            vChop = video[d0:d1,y0:y1,x0:x1]
            vpp.videoPlay(vChop)
            vChop = vpp.videoProcess(vChop, (112,128,3), downSample = 2, NormEn=True, RLFlipEn=False)
            vpp.videoPlay(vpp.videoFormat(vChop))
            vChop = np.reshape(vChop,(3,1,16,112,128,3))
            pred_y = np.argmax(c3d.evaluateProb(vChop, sess))
            print('The predicted lable is ', pred_y)
            pred_yList.append(pred_y)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)