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
    frmSize = (112,80,3)
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize, nof_conv1= 32, nof_conv2=128, nof_conv3=256, nof_conv4= 512, noo_fc6=4096, noo_fc7=4096)
    
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
        ut_set = ut.ut_interaction_set2_a(frmSize)
        seqRange = range(11,21)
        savePrefix = 'c3d_finetune_on_ut_single_net_set2_'
        log = time.ctime() + ' Finetune the 3D-ConvNet on UT-Interaction dataset set2! \n'
    else:    
        ut_set = ut.ut_interaction_set1_a(frmSize)
        seqRange = range(1,11)
        savePrefix = 'c3d_finetune_on_ut_single_net_set1_'
        log = time.ctime() + ' Finetune the 3D-ConvNet on UT-Interaction dataset set1! \n'
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    iteration = 4001
    batchSize = 15
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
        saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
        #saver_feature_g.restore(sess,join(common.path.variablePath, 'c3d_pretrain_on_ucf_fg.ckpt'))
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' + \
              '****************************************\n'
        common.pAndWf(logName,log)
        ut_set.splitTrainingTesting(seq)
        ut_set.loadTrainingAll(oneHotLabelMode=True)
        test_x,test_y = ut_set.loadTesting(oneHotLabelMode=True)
        if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
            best_accuracy = 0
            anvAccuList = np.zeros((10))
            for i in range(iteration):
                train_x,train_y = ut_set.loadTrainingBatch(batchSize)
                c3d.train(train_x, train_y, sess)
                if i%int(iteration/200) == 0:
                    train_accuracy,_ = c3d.top2Accu(train_x, train_y, sess)
                    test_accuracy,t2y_accu = c3d.top2Accu(test_x, test_y, sess)
                    anvAccuList = np.append(anvAccuList[1:10],test_accuracy)
                    anv_accuracy = np.mean(anvAccuList)
                    if anv_accuracy > best_accuracy:
                        best_accuracy = anv_accuracy
                    log = "step %d, training: %g, testing: %g, anv: %g, best %g \n"%(i, train_accuracy, test_accuracy, anv_accuracy, best_accuracy)
                    common.pAndWf(logName,log)
                    if anv_accuracy == 1 or (i > int(iteration * 0.75) and anv_accuracy >= best_accuracy):
                        break
            save_path = saver.save(sess,join(common.path.variablePath, savePrefix  + str(seq) +'.ckpt'))
            common.pAndWf(logName,' \n')
        else:
            variableName = savePrefix + str(seq) + '.ckpt'
            saver.restore(sess,join(common.path.variablePath, variableName))
            # begin to test
            test_accuracy = c3d.test(test_x, test_y, sess)
            log = "Testing accuracy %g \n"%(test_accuracy)
            common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)