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

def loadTrainBatch(dataset,n,q):
    trainSet = dataset.loadTrainBatchMP(n)
    q.put(trainSet)

def main(_):
    # ******************************************************
    # define the network
    # ******************************************************
    numOfClasses = 30 
    frmSize = (112,128,3)
    with tf.device('/gpu:0'):
        with tf.variable_scope('atomic_action_features') as scope:
            c3d = network.C3DNET(numOfClasses, frmSize)
    
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
    test_x,test_y = ucf_set.loadTest(20) 
    print('initial testing accuracy ',c3d.evaluate(test_x, test_y, sess))
   
    # ******************************************************
    # Train and test the network 
    # ******************************************************
    logName = 'c3d_pretrain_on_ucf.txt'
    common.clearFile(logName)
    iteration = 20001 
    batchSize = 5 
    best_accuracy = 0
    q = multiprocessing.Queue()
    pro_loadTrain = multiprocessing.Process(target=loadTrainBatch,args=(ucf_set,batchSize,q))
    for i in range(iteration):
        pro_loadTrain.start()
        train_x,train_y = q.get()
        print(trian_y)
        if i%int(iteration/200) == 0:
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