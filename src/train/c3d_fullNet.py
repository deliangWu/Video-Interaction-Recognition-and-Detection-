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
import time

def main(argv):
    # ***********************************************************
    # define the network
    # ***********************************************************
    numOfClasses = 6 
    frmSize = [(112,128,3),(112,80,3)]
    
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET_3F1C(numOfClasses, frmSize)
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
    if len(argv) >= 3 and argv[2] == 'set2':
        ut_set = ut.ut_interaction_set2_ga(frmSize)
        seqRange = range(11,21)
        savePrefix = 'c3d_fullNet_ut_set2_'
        log = time.ctime() + ' Train the 3D-ConvNet full Net on UT-Interaction dataset set2 from scratch! \n'
    else:    
        ut_set = ut.ut_interaction_set1_ga(frmSize)
        seqRange = range(1,11)
        savePrefix = 'c3d_fullNet_ut_set1_'
        log = time.ctime() + ' Train the 3D-ConvNet full Net on UT-Interaction dataset set1 from scratch! \n'
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    common.pAndWf(logName,log)    
    iteration = 4001
    batchSize = 15
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
        saver_feature_a0 = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_a0_VarsList])
        saver_feature_a1 = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_a1_VarsList])
        saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_3f1c_VarsList])
        saver_feature_g.restore(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fg.ckpt'))
        saver_feature_a0.restore(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fa0.ckpt'))
        saver_feature_a1.restore(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fa1.ckpt'))
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        ut_set.splitTrainingTesting(seq)
        test_x,test_x0, test_x1, test_y = ut_set.loadTesting()
        if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
            best_accuracy = 0
            anvAccuList = np.zeros((10))
            for i in range(iteration):
                train_x,train_x0, train_x1, train_y = ut_set.loadTrainingBatch(batchSize)
                if i%int(iteration/200) == 0:
                    train_accuracy = c3d.test(train_x, train_x0, train_x1, train_y, sess)
                    test_accuracy = c3d.test(test_x, test_x0, test_x1, test_y, sess)
                    anvAccuList = np.append(anvAccuList[1:10],test_accuracy)
                    anv_accuracy = np.mean(anvAccuList)
                    if anv_accuracy > best_accuracy:
                        best_accuracy = anv_accuracy
                    log = "step %d, training: %g, testing: %g, anv: %g, best %g \n"%(i, train_accuracy, test_accuracy, anv_accuracy, best_accuracy)
                    common.pAndWf(logName,log)
                    if anv_accuracy == 1 or (i > int(iteration * 0.75) and anv_accuracy >= best_accuracy):
                        saver_classifier.save(sess,join(common.path.varibelPath, savePrefix+str(seq)+'_3f1c.ckpt'))
                        break
                    
                c3d.train(train_x, train_x0, train_x1, train_y, sess)
            common.pAndWf(logName,' \n')
        else:
            # begin to test
            test_accuracy = c3d.test(test_x, test_y, sess)
            log = "Testing accuracy %g \n"%(test_accuracy)
            common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)