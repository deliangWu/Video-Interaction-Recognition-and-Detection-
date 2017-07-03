'''a 7-class singleNet model for interaction detector'''
from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import sys
from os.path import join 
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
sys.path.insert(1,'../train')
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
    numOfClasses = 7 
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
            ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=7)
            savePrefix = 'c3d_c7_det_set1_'
            log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
        else:
            ut_set = ut.ut_interaction_set2(frmSize,numOfClasses=7)
            savePrefix = 'c3d_c7_det_set2_'
            log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    elif len(argv) >= 3 and argv[2] == 'set2':
        seqRange = range(11,21)
        ut_set = ut.ut_interaction_set2(frmSize,numOfClasses=7)
        savePrefix = 'c3d_c7_det_set2_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set2 from scratch! \n'
    else:
        seqRange = range(1,11)
        ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=7)
        savePrefix = 'c3d_c7_det_set1_'
        log = time.ctime() + ' Train the 3D-ConvNet on UT-Interaction dataset set1 from scratch! \n'
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    saver_net = tf.train.Saver()
    logName =  savePrefix + common.getDateTime() + '.txt'
    common.clearFile(logName)
    common.pAndWf(logName,log)    
    iteration = 2001
    batchSize = 16 
    for run in range(1): 
        log = '-------------------------------------------------------------------------\n' \
            + '---------------------------- RUN ' + str(run) + ' ------------------------------\n' \
            + '-------------------------------------------------------------------------\n'
        common.pAndWf(logName,log)
        accuSet = []
        t2accuSet = []
        # 10-flod
        for seq in seqRange:
            with sess.as_default():
                sess.run(initVars)
            log = '****************************************\n' \
                + 'current sequence is ' + str(seq)  + '\n' + \
                  '****************************************\n'
            common.pAndWf(logName,log)
            # load evaluating videos
            ut_set.splitTrainingTesting(seq, loadTrainingEn=False)
            test_x,test_y = ut_set.loadTesting(oneHotLabelMode=True,downSample = 2)
            if len(argv) < 2 or argv[1] == 'train' or argv[1] == 'Train':
                f_var = join(common.path.variablePath, savePrefix + str(seq) + '_det7c.ckpt')
                # load pre-trained parameters if there are pre-trained parameters
                if os.path.isfile(join(common.path.variablePath, savePrefix + str(seq) + '_det7c.ckpt.index')):
                    print('load pre-trained variables!')
                    saver_net.restore(sess,f_var)
                else:
                    print('Train the model from scratch!')
                # load training videos
                ut_set.loadTrainingAll(oneHotLabelMode=True,downSample=2)
                best_accuracy = 0
                epoch = 0
                anvAccuList = np.zeros((3))
                i = 0
                while True:
                    # load training mini-batch
                    train_x,train_y = ut_set.loadTrainingBatch(batchSize)
                    
                    # get training epoch
                    epoch = ut_set.getEpoch()
                    
                    # set learning rate
                    #learning_rate = 0.0001 * 2**(-int(epoch/8))
                    learning_rate = 0.0001 * 2**(-int(epoch/4))
                    #learning_rate = 0.0001
                    
                    # execute training 
                    c3d.train(train_x, train_y, sess, learning_rate=learning_rate)

                    # observe the training and evaluating accuracy and loss                    
                    if i% 10 == 0:
                        # training accuracy and loss
                        train_accuracy,_ = c3d.top2Accu(train_x, train_y, sess)
                        loss_tr = c3d.getLoss(train_x, train_y, sess)
                        # evaluating accuracy and loss
                        test_accuracy,t2y_accu = c3d.top2Accu(test_x, test_y, sess)
                        loss_t = c3d.getLoss(test_x, test_y, sess)
                        
                        # get anverage evaluating accuracy                        
                        anvAccuList = np.append(anvAccuList[1:3],test_accuracy)
                        anv_accuracy = np.mean(anvAccuList)
                        if anv_accuracy > best_accuracy:
                            best_accuracy = anv_accuracy
                        log = "seq: %d, epoch: %d, step: %d, training: %g, testing: %g, loss_tr: %g, loss_t: %g  \n"%(seq, epoch, i, train_accuracy, test_accuracy, loss_tr, loss_t)
                        common.pAndWf(logName,log)
                        if i >= 1000:
                            break
                    i+=1
                # save trained parameters
                saver_net.save(sess,join(common.path.variablePath, savePrefix + str(seq) + '_det7c.ckpt'))
                common.pAndWf(logName,' \n')
            # execute testing
            else:
                saver_net.restore(sess,join(common.path.variablePath, savePrefix  + str(seq) + '_det7c.ckpt'))
                test_accuracy,t2y_accu = c3d.top2Accu(test_x, test_y, sess)
                print('testing accu is: ',test_accuracy, ' and top2 accu is ', t2y_accu)
                
            
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)