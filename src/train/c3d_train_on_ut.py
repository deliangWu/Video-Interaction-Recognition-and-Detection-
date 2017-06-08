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
    with tf.variable_scope('top') as scope:
        c3d = network.C3DNET(numOfClasses, frmSize,nof_conv1=32, nof_conv2= 128, nof_conv3=256, nof_conv4= 512, noo_fc6=4096, noo_fc7=4096)
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
    if len(argv) >= 3 and argv[2][0:3] == 'seq':
        seqNo = int(argv[2][3:])
        seqRange = (seqNo,)
        if seqNo <= 10:
            ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=numOfClasses)
            savePrefix = 'c3d_train_on_ut_set1_'
            log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
        else:
            ut_set = ut.ut_interaction_set2(frmSize,numOfClasses=numOfClasses)
            savePrefix = 'c3d_train_on_ut_set2_'
            log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    elif len(argv) >= 3 and argv[2] == 'set2':
        seqRange = range(11,21)
        ut_set = ut.ut_interaction_set2(frmSize,numOfClasses=numOfClasses)
        savePrefix = 'c3d_train_on_ut_set2_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    else:
        seqRange = range(1,11)
        ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=numOfClasses)
        savePrefix = 'c3d_train_on_ut_set1_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
    saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    common.pAndWf(logName,log)    
    iteration = 2001
    batchSize = 16 
    for run in range(10): 
        log = '-------------------------------------------------------------------------\n' \
            + '---------------------------- RUN ' + str(run) + ' ------------------------------\n' \
            + '-------------------------------------------------------------------------\n'
        common.pAndWf(logName,log)
        accuSet = []
        t2accuSet = []
        for seq in seqRange:
            with sess.as_default():
                sess.run(initVars)
            log = '****************************************\n' \
                + 'current sequence is ' + str(seq)  + '\n' + \
                  '****************************************\n'
            common.pAndWf(logName,log)
            ut_set.splitTrainingTesting(seq, loadTrainingEn=False)
            ut_set.loadTrainingAll(oneHotLabelMode=True)
            test_x,test_y = ut_set.loadTesting(oneHotLabelMode=True)
            if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
                best_accuracy = 0
                epoch = 0
                anvAccuList = np.zeros((3))
                loss_t_list = np.zeros((5)) + 1e10
                i = 0
                while True:
                    train_x,train_y = ut_set.loadTrainingBatch(batchSize)
                    epoch = ut_set.getEpoch()
                    #learning_rate = 0.0001 * 2**(-int(epoch/8))
                    learning_rate = 0.0001 * 2**(-int(epoch/4))
                    #learning_rate = 0.0001
                    c3d.train(train_x, train_y, sess, learning_rate=learning_rate)
                    #loss = c3d.getLoss(train_x, train_y, sess)
                    #print('step: %d, loss: %g '%(i,loss))
                    if i% 10 == 0:
                        train_accuracy,_ = c3d.top2Accu(train_x, train_y, sess)
                        loss_tr = c3d.getLoss(train_x, train_y, sess)
                        loss_t = c3d.getLoss(test_x, test_y, sess)
                        test_accuracy,t2y_accu = c3d.top2Accu(test_x, test_y, sess)
                        pre_loss_t_mean = np.mean(loss_t_list)
                        loss_t_list = np.append(loss_t_list[1:],loss_t)
                        loss_t_mean = np.mean(loss_t_list)
                        
                        #c3d.obs(test_x, test_y, sess)
                        #anvAccuList = np.append(anvAccuList[1:3],test_accuracy)
                        #anv_accuracy = np.mean(anvAccuList)
                        #if anv_accuracy > best_accuracy:
                        #    best_accuracy = anv_accuracy
                        log = "seq: %d, epoch: %d, step: %d, training: %g, loss_tr: %g, loss_t: %g, testing: %g, t2y: %g\n"%(seq, epoch, i, train_accuracy, loss_tr,loss_t, test_accuracy, t2y_accu)
                        common.pAndWf(logName,log)
                        if test_accuracy == 1 or loss_t_mean - pre_loss_t_mean > 20 or i > 500:
                            break
                    i+=1
                saver_feature_g.save(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_fg6.ckpt'))
                saver_classifier.save(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_c6.ckpt'))
                common.pAndWf(logName,' \n')
            accuSet.append(test_accuracy)
            t2accuSet.append(t2y_accu)
        log = 'The list of Classification Accuracy: ' + str(accuSet) + \
              '\n ' + str(t2accuSet) + \
              '\n Mean Classification Accuracy is ' + str(np.mean(accuSet)) + ', and top2 mean accuracy is ' + str(np.mean(t2accuSet)) + '\n' + \
              '__________________________________________________________________________________________________\n \n'
        common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)