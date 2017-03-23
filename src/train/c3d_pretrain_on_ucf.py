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
    iteration = 1301 
    batchSize = 15
    
    # define the network
    c3d = network.C3DNET(numOfClasses, frmSize)
    
    # define session
    sess = tf.InteractiveSession()
    initVars = tf.global_variables_initializer()
    sess.run(initVars)
    test_x,test_y = ucf_set.loadTest(50) 
    ucf_set.loadTrainAll()
    for i in range(iteration):
        train_x,train_y = ucf_set.loadTrainBatch(batchSize)
        if i%int(iteration/20) == 0:
            train_accuracy = c3d.evaluate(train_x, train_y, sess)
            test_accuracy = c3d.evaluate(test_x, test_y, sess)
            log = "step %d, training accuracy %g and testing accuracy %g \n"%(i, train_accuracy, test_accuracy)
            common.pAndWf(logName,log)
        c3d.train(train_x, train_y, sess)
if __name__ == "__main__":
    tf.app.run(main=main)