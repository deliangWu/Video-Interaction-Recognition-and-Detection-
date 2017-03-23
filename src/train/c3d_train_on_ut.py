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

def main(argv):
    # define the dataset
    numOfClasses = 6 
    frmSize = (112,128,3)
    ut_set = ut.ut_interaction_set1(frmSize)
    seqRange = range(1,2)
    logName = 'c3d_train_on_ut_set1.txt'
    common.clearFile(logName)
    iteration = 2001
    batchSize = 15
    
    # define the network
    with tf.variable_scope('global_interaction_features') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize)
    
    # define session
    sess = tf.InteractiveSession()
    initVars = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    for seq in seqRange:
        log = '**************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        with sess.as_default():
            sess.run(initVars)
        ut_set.splitTrainingTesting(seq)
        test_x,test_y = ut_set.loadTesting_new()
        if argv[1] == 'train' or argv[1] == 'Train':
            best_accuracy = 0
            for i in range(iteration):
                train_x,train_y = ut_set.loadTraining(batchSize)
                if i%int(iteration/100) == 0:
                    train_accuracy = c3d.evaluate(train_x, train_y, sess)
                    test_accuracy = c3d.evaluate(test_x, test_y, sess)
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
            saver.restore(sess,join(common.path.variablePath, 'c3d_train_on_ut_' + str(seq) + '.ckpt'))
        test_accuracy = c3d.evaluate(test_x, test_y, sess)
        log = "Testing accuracy %g \n"%(test_accuracy)
        common.pAndWf(logName,log)
            
        # begin to test
        
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)