
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
        if len(argv) >= 3 and argv[2] == 'unShareFeatureVariable':
            savePrefix = 'c3d_finetune_on_ut_dual_nets_unShareVars_'
            log = time.ctime() + ' Finetune the dual-nets model with two independent feature variables on UT-Interaction '
            c3d = network.C3DNET_2F1C(numOfClasses, frmSize, shareFeatureVariable= False)
        else:
            savePrefix = 'c3d_finetune_on_ut_dual_nets_shareVars_'
            log = time.ctime() + ' Finetune the dual-nets model with two sharing feature variables on UT-Interaction '
            c3d = network.C3DNET_2F1C(numOfClasses, frmSize)
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
    if len(argv) >= 4 and argv[3] == 'set2':
        ut_set = ut.ut_interaction_set2_atomic(frmSize)
        seqRange = range(11,21)
        savePrefix = savePrefix + 'set2_'
        log = log + 'set2! \n'
    else:    
        ut_set = ut.ut_interaction_set1_atomic(frmSize)
        seqRange = range(1,3)
        savePrefix = savePrefix + 'set1_'
        log = log + 'set1! \n'
    
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    common.pAndWf(logName,log)
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    iteration = 4001
    batchSize = 16 
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        saver_feature_a0 = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_a0_VarsList])
        saver_feature_a1 = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_a1_VarsList])
        saver_classifier_2f1c = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_2f1c_VarsList])
        log = '****************************************\n' \
            + 'current sequence is ' + str(seq)  + '\n' +  \
              '****************************************\n'
        common.pAndWf(logName,log)
        ut_set.splitTrainingTesting(seq)
        test_x0,test_x1,test_y = ut_set.loadTesting()
        if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
            ut_set.loadTrainingAll()
            best_accuracy = 0
            anvAccuList = np.zeros((3))
            for i in range(iteration):
                train_x0,train_x1,train_y = ut_set.loadTrainingBatch(batchSize)
                epoch = ut_set.getEpoch()
                if i%int(iteration/200) == 0:
                    train_accuracy = c3d.test(train_x0, train_x1, train_y, sess)
                    test_accuracy = c3d.test(test_x0, test_x1, test_y, sess)
                    anvAccuList = np.append(anvAccuList[1:3],test_accuracy)
                    anv_accuracy = np.mean(anvAccuList)
                    if anv_accuracy > best_accuracy:
                        best_accuracy = anv_accuracy
                    log = "Epoch: %d, step: %d, training: %g, testing: %g, anv: %g, best: %g \n"%(epoch, i, train_accuracy, test_accuracy, anv_accuracy, best_accuracy)
                    common.pAndWf(logName,log)
                    if anv_accuracy == 1 or (i > int(iteration * 0.75) and anv_accuracy >= best_accuracy):
                        break
                learning_rate = 0.0005 * 2**(-int(epoch/6))
                c3d.train(train_x0, train_x1, train_y, sess, learning_rate=learning_rate)
            save_path_fa0 = saver_feature_a0.save(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fa0.ckpt'))
            save_path_fa1 = saver_feature_a1.save(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fa1.ckpt'))
            save_path_2f1c = saver_classifier_2f1c.save(sess,join(common.path.variablePath, savePrefix + str(seq) +'_2f1c.ckpt'))
            common.pAndWf(logName,' The training is finished at ' + time.ctime() + ' \n')
        else:
            # begin to test
            saver_feature_a0.restore(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fa0.ckpt'))
            saver_feature_a1.restore(sess,join(common.path.variablePath, savePrefix + str(seq) +'_fa1.ckpt'))
            saver_classifier_2f1c.restore(sess,join(common.path.variablePath, savePrefix + str(seq) +'_2f1c.ckpt'))
            test_accuracy = c3d.test(test_x0, test_x1, test_y, sess)
            testProb = c3d.evaluateProb(test_x0,test_x1,sess)
            print(testProb)
            print(test_y)
            log = "Testing accuracy %g \n"%(test_accuracy)
            common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)