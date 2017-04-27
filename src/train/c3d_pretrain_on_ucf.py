import numpy as np
import os
from os.path import isfile, join
import sys
import tensorflow as tf
import multiprocessing
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
sys.path.insert(1,'../common')
import ut_interaction as ut
import ucf101 as ucf 
import videoPreProcess as vpp
import model
import common
import network
import time

def main(argv):
    # ******************************************************
    # define the network
    # ******************************************************
    numOfClasses = 20 
    frmSize = (112,128,3)
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize, nof_conv1= 64, nof_conv2=128, nof_conv3=256, nof_conv4= 256, nof_conv5=256)
    
    # ******************************************************
    # define session
    # ******************************************************
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver()
    initVars = tf.global_variables_initializer()
    with sess.as_default():
        sess.run(initVars)
   
    # ******************************************************
    # load the dataset into memory
    # ******************************************************
    ucf_set = ucf.ucf101(frmSize,numOfClasses) 
    test_x,test_y = ucf_set.loadTesting(numOfProcesses = 4) 
    print('initial testing accuracy ',c3d.test(test_x, test_y, sess))
   
    # ******************************************************
    # Train and test the network 
    # ******************************************************
    saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
    saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
    logName = 'c3d_pretrain_on_ucf_' + common.getDateTime() + '.txt'
    variableName = 'c3d_pretrain_on_ucf.ckpt'
    common.clearFile(logName)
    iteration = 20001 
    batchSize = 30
    best_accuracy = 0
    if len(argv) < 2 or argv[1] == 'Train' or argv[1] == 'train':
        print('Start to loading videos for training..................')
        ucf_set.loadTrainingAll(numOfProcesses = 4)
        for i in range(iteration):
            train_x,train_y = ucf_set.loadTrainBatch(batchSize) 
            if i%int(iteration/200) == 0:
                train_accuracy = c3d.test(train_x, train_y, sess)
                test_accuracy = c3d.test(test_x, test_y, sess)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                log = "step %d, training accuracy %g and testing accuracy %g , best accuracy is %g \n"%(i, train_accuracy, test_accuracy, best_accuracy)
                common.pAndWf(logName,log)
                if (test_accuracy == 1) or (i > int(iteration*0.75) and test_accuracy >= best_accuracy):
                    break
            epoch = ucf_set.getTrainingEpoch()
            learning_rate = 0.005 * 10**(-int(epoch/4))
            c3d.train(train_x, train_y, sess,learning_rate=learning_rate)
        saver_feature_g.save(sess,join(common.path.variablePath, 'c3d_pretrain_on_ucf_fg.ckpt'))
        saver_classifier.save(sess,join(common.path.variablePath,'c3d_pretrain_on_ucf_c.ckpt'))
    else:
        variableName = 'c3d_pretrain_on_ucf_0329.ckpt'
        saver.restore(sess,join(common.path.variablePath, variableName))
        # begin to test
        test_accuracy = c3d.test(test_x, test_y, sess)
        log = "Testing accuracy %g \n"%(test_accuracy)
        common.pAndWf(logName,log)
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)