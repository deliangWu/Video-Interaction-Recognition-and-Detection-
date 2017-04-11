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
    numOfClasses = 2 
    frmSize = (112,128,3)
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize,nof_conv1=64, nof_conv2= 128, nof_conv3=256, nof_conv4= 256, nof_conv5=256)
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
        ut_set = ut.ut_interaction_set2(frmSize)
        seqRange = range(11,21)
        savePrefix = 'c3d_train_on_ut_set2_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    else:    
        ut_set = ut.ut_interaction_set1(frmSize)
        #seqRange = range(1,11)
        seqRange = (1,4,10)
        savePrefix = 'c3d_train_on_ut_set1_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    common.pAndWf(logName,log)    
    iteration = 1001
    batchSize = 16 
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
        saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
        #saver_feature_g.restore(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_fg.ckpt'))
        #saver_classifier.restore(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_c.ckpt'))
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        ut_set.splitTrainingTesting(seq, loadTrainingEn=False)
        ut_set.loadTrainingAll()
        test_x,test_lable = ut_set.loadTesting()
        for testlabel in range(6):
            print('test lable ----- ',testlabel)
            test_y = ut.oneVsRest(test_lable,testlabel)
            print(test_y)
            if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
                best_accuracy = 0
                anvAccuList = np.zeros((3))
                for i in range(iteration):
                    train_x,train_y = ut_set.loadTrainingBatch(batchSize)
                    train_y = ut.oneVsRest(train_y,testlabel)
                    print(train_y)
                    
                    epoch = ut_set.getEpoch()
                    if i%int(iteration/50) == 0:
                        train_accuracy = c3d.test(train_x, train_y, sess)
                        test_accuracy = c3d.test(test_x, test_y, sess)
                        anvAccuList = np.append(anvAccuList[1:3],test_accuracy)
                        anv_accuracy = np.mean(anvAccuList)
                        if anv_accuracy > best_accuracy:
                            best_accuracy = anv_accuracy
                        log = "epoch: %d, step: %d, training: %g, testing: %g, anv: %g, best: %g \n"%(epoch, i, train_accuracy, test_accuracy, anv_accuracy, best_accuracy)
                        common.pAndWf(logName,log)
                        if anv_accuracy == 1 or (i > int(iteration * 0.75) and anv_accuracy >= best_accuracy):
                            break
                    learning_rate = 0.005 * 2**(-int(epoch/4))
                    c3d.train(train_x, train_y, sess, learning_rate=learning_rate)
                #saver_feature_g.save(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_fg.ckpt'))
                #saver_classifier.save(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_c.ckpt'))
                common.pAndWf(logName,' \n')
        else:
            saver_feature_g.restore(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_fg.ckpt'))
            saver_classifier.restore(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_c.ckpt'))
            # begin to test
            test_accuracy = c3d.test(test_x, test_y, sess)
            test_prob = c3d.evaluateProb(test_x,sess)
            print('test_prob is \n', test_prob, '\n \n', \
                  'test_y is \n', test_y)
            log = "Testing accuracy %g \n"%(test_accuracy)
            common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)