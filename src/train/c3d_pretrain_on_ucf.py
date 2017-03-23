import numpy as np
import os
from os.path import isfile, join
import tensorflow as tf
import sys
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
sys.path.insert(1,'../common')
import ut_interaction as ut
import ucf101 as ucf 
import videoPreProcess as vpp
import model
import common
import network

def main(_):
    # define the dataset
    numOfClasses = 20 
    frmSize = (112,80,3)
    ucf_set = ucf.ucf101(frmSize,numOfClasses) 
    logName = 'c3d_pretrain_on_ucf.txt'
    common.clearFile(logName)
    iteration = 20001 
    batchSize = 15
    
    # define the network
    with tf.variable_scope('atomic_action_features') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize)
    
    # define session
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    initVars = tf.global_variables_initializer()
    sess.run(initVars)
    test_x,test_y = ucf_set.loadTest(50) 
    ucf_set.loadTrainAll()
    for i in range(iteration):
        train_x,train_y = ucf_set.loadTrainBatch(batchSize)
        best_accuracy = 0
        if i%int(iteration/20) == 0:
            train_accuracy = c3d.evaluate(train_x, train_y, sess)
            test_accuracy = c3d.evaluate(test_x, test_y, sess)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            log = "step %d, training accuracy %g and testing accuracy %g , best accuracy is %g \n"%(i, train_accuracy, test_accuracy, best_accuracy)
            common.pAndWf(logName,log)
            if (test_accuracy == 1) or (i > int(iteration*0.75) and test_accuracy >= best_accuracy):
                save_path = saver.save(sess,join(common.path.variablePath, 'c3d_pretrain_on_ucf.ckpt'))
                break
        c3d.train(train_x, train_y, sess)
if __name__ == "__main__":
    tf.app.run(main=main)