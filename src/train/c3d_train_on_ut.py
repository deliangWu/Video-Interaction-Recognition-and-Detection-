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
    frmSize = (112,128,3)
    with tf.variable_scope('global_interaction_features') as scope:
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
    ut_set = ut.ut_interaction_set2(frmSize)
    seqRange = range(11,21)
    logName = 'c3d_train_on_ut_' + common.getDateTime() + '.txt'
    common.clearFile(logName)
    log = time.ctime() + 'Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    common.pAndWf(logName,log)
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    iteration = 2001
    batchSize = 15
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        saver = tf.train.Saver()
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        ut_set.splitTrainingTesting(seq)
        test_x,test_y = ut_set.loadTesting()
        if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
            best_accuracy = 0
            for i in range(iteration):
                train_x,train_y = ut_set.loadTrainingBatch(batchSize)
                if i%int(iteration/100) == 0:
                    train_accuracy = c3d.test(train_x, train_y, sess)
                    test_accuracy = c3d.test(test_x, test_y, sess)
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                    log = "step %d, training accuracy %g and testing accuracy %g, best accuracy is %g \n"%(i, train_accuracy, test_accuracy, best_accuracy)
                    common.pAndWf(logName,log)
                    if test_accuracy == 1 or (i > int(iteration * 0.75) and test_accuracy >= best_accuracy):
                        save_path = saver.save(sess,join(common.path.variablePath, 'c3d_train_on_ut_' + str(seq) +'.ckpt'))
                        break
                c3d.train(train_x, train_y, sess)
            common.pAndWf(logName,' \n')
        else:
            variableName = 'c3d_train_on_ut_' + common.getDateTime() + '_' + str(seq) + '.ckpt'
            saver.restore(sess,join(common.path.variablePath, variableName))
            # begin to test
            test_accuracy = c3d.test(test_x, test_y, sess)
            log = "Testing accuracy %g \n"%(test_accuracy)
            common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)