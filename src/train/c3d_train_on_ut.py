import numpy as np
import os
import tensorflow as tf
import sys
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
sys.path.insert(1,'../common')
import ut_interaction as ut
import model
import common
import network

def main(_):
    # define the dataset
    numOfClasses = 6 
    frmSize = (112,128,3)
    ut_set = ut.ut_interaction_set1(frmSize)
    logName = 'c3d_train_on_set1.txt'
    common.clearFile(logName)
    seqRange = range(1,2)
    iteration = 1301
    batchSize = 15
    
    # define the network
    c3d = network.C3DNET(numOfClasses, frmSize)
    
    # define session
    sess = tf.InteractiveSession()
    initVars = tf.global_variables_initializer()
    
    for seq in seqRange:
        log = '**************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        sess.run(initVars)
        ut_set.splitTrainingTesting(seq)
        test_x,test_y = ut_set.loadTesting_new()
        for i in range(iteration):
            train_x,train_y = ut_set.loadTraining(batchSize)
            if i%int(iteration/20) == 0:
                train_accuracy = c3d.evaluate(train_x, train_y, sess)
                test_accuracy = c3d.evaluate(test_x, test_y, sess)
                log = "step %d, training accuracy %g and testing accuracy %g \n"%(i, train_accuracy, test_accuracy)
                common.pAndWf(logName,log)
            c3d.train(train_x, train_y, sess)
        common.pAndWf(logName,' \n')
if __name__ == "__main__":
    tf.app.run(main=main)