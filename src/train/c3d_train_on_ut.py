'''run training and testing of the singleNet model, a singleNet model is consist of data pre-processing, feature descriptor(only global feature descriptor), classifier and optimizer. '''

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
            ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=numOfClasses,c3En=True)
            savePrefix = 'c3d_train_on_ut_set1_'
            log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
        else:
            ut_set = ut.ut_interaction_set2(frmSize,numOfClasses=numOfClasses,c3En=True)
            savePrefix = 'c3d_train_on_ut_set2_'
            log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    elif len(argv) >= 3 and argv[2] == 'set2':
        seqRange = range(11,21)
        ut_set = ut.ut_interaction_set2(frmSize,numOfClasses=numOfClasses,c3En=True)
        savePrefix = 'c3d_train_on_ut_set2_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    else:
        seqRange = range(1,11)
        ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=numOfClasses, c3En=True)
        savePrefix = 'c3d_train_on_ut_set1_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    #saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
    #saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
    saver_net = tf.train.Saver()
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    common.pAndWf(logName,log)    
    iteration = 2001
    batchSize = 16 
    # 10 runs to get an average performance. 
    for run in range(10): 
        log = '-------------------------------------------------------------------------\n' \
            + '---------------------------- RUN ' + str(run) + ' ------------------------------\n' \
            + '-------------------------------------------------------------------------\n'
        common.pAndWf(logName,log)
        accuSet = []
        t2accuSet = []
        # 10-fold cross validation
        for seq in seqRange:
            with sess.as_default():
                sess.run(initVars)
            log = '****************************************\n' \
                + 'current sequence is ' + str(seq)  + '\n' + \
                  '****************************************\n'
            common.pAndWf(logName,log)
            ut_set.splitTrainingTesting(seq, loadTrainingEn=False)
            test_x,test_y = ut_set.loadTesting(oneHotLabelMode=True)
            if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
                ut_set.loadTrainingAll(oneHotLabelMode=True)
                best_accuracy = 0
                epoch = 0
                anvAccuList = np.zeros((3))
                loss_t_list = np.zeros((3)) + 1e10
                loss_tr_list = np.zeros((3)) + 1e10
                loss_t_min = 1e10
                loss_tr_min = 1e10
                i = 0
                while True:
                    # load traning video batch                    
                    train_x,train_y = ut_set.loadTrainingBatch(batchSize)
                    
                    # get current training epoch
                    epoch = ut_set.getEpoch()
                    
                    # set learning rate                    
                    #learning_rate = 0.0001 * 2**(-int(epoch/3))
                    learning_rate = 1e-4
                    #learning_rate = 0.1 * 2**(-int(epoch/4))
                    
                    # excute training                    
                    c3d.train(train_x, train_y, sess, learning_rate=learning_rate)

                    # observe the training and evaluating performance (accuracy and loss) while training                    
                    if i% 10 == 0:
                        # get training accuracy and loss
                        train_accuracy,_ = c3d.top2Accu(train_x, train_y, sess)
                        loss_tr = c3d.getLoss(train_x, train_y, sess)
                        # get evaluating accuracy and loss
                        test_accuracy,t2y_accu = c3d.top2Accu(test_x, test_y, sess)
                        loss_t = c3d.getLoss(test_x, test_y, sess)
                        
                        # anverage training and evaluating accuracy and loss
                        if i == 0:
                            anvAccuList = np.array([test_accuracy]*3)
                            loss_t_list = np.array([loss_t]*3)
                            loss_tr_list = np.array([loss_tr]*3)
                        else:
                            anvAccuList = np.append(anvAccuList[1:],test_accuracy)
                            loss_t_list = np.append(loss_t_list[1:],loss_t)
                            loss_tr_list = np.append(loss_tr_list[1:],loss_tr)
                        anv_accuracy = np.mean(anvAccuList)
                        loss_t_mean = np.mean(loss_t_list)
                        loss_tr_mean = np.mean(loss_tr_list)
                        
                        if loss_t_mean < loss_t_min:
                            loss_t_min = loss_t_mean
                        
                        log = "seq%d, epoch%d, step: %d, training: %g, loss_tr: %g, loss_t: %g, testing: %g, t2y: %g\n"%(seq, epoch, i, train_accuracy, loss_tr_mean,loss_t_mean, anv_accuracy, t2y_accu)
                        common.pAndWf(logName,log)
                        if i > 700:
                            break
                    i+=1
                # save the trained network
                #saver_feature_g.save(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_fg6.ckpt'))
                #saver_classifier.save(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_c6.ckpt'))
                saver_net.save(sess,join(common.path.variablePath2, savePrefix  + str(seq) + '.ckpt'))
                common.pAndWf(logName,' \n')
            # run testing
            else:
                saver_net.restore(sess, join(common.path.variablePath2, savePrefix  + str(seq) + '.ckpt'))
                test_accuracy,t2y_accu = c3d.top2Accu(test_x, test_y, sess)
                print('Testing accuracy is ', test_accuracy)
            accuSet.append(test_accuracy)
            t2accuSet.append(t2y_accu)
        log = 'The list of Classification Accuracy: ' + str(accuSet) + \
                      '\n ' + str(t2accuSet) + \
                      '\n Mean Classification Accuracy is ' + str(np.mean(accuSet)) + ', and top2 mean accuracy is ' + str(np.mean(t2accuSet)) + '\n' + \
                      '__________________________________________________________________________________________________\n \n'
        common.pAndWf(logName,log)
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)