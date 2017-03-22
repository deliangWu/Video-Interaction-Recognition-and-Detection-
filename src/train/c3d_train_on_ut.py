import numpy as np
import os
import tensorflow as tf
import sys
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../model')
import ut_interaction as ut
import model

def main(_):
    numOfClasses = 6 
    frmSize = (112,128,3)
    # define the dataset
    ut_set1 = ut.ut_interaction_set1(frmSize)

    # build the 3D ConvNet
    # define the input and output variables
    x = tf.placeholder(tf.float32, (None,16) + frmSize)
    y_ = tf.placeholder(tf.float32, (None, numOfClasses))
    featuresT = tf.placeholder(tf.float32,(None,4096))
    keep_prob = tf.placeholder(tf.float32)
    
    features = model.FeatureDescriptor.c3d(x,frmSize,keep_prob)
    vars_dict = {
                "W_sm":model.weight_variable([4096,numOfClasses]),
                "b_sm":model.bias_variable([numOfClasses])
                 }
    y_conv = model.Classifier.softmax(features,vars_dict)
    y_convT = model.Classifier.softmax(featuresT,vars_dict)
    
    # Train and evaluate the model
    sess = tf.InteractiveSession()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    correct_predictionT = tf.equal(tf.argmax(y_convT,1), tf.argmax(y_,1))
    accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
    
    # Create a saver
    saver = tf.train.Saver()
    with sess.as_default():
        if os.name == 'nt':
            devs = ['/cpu:0']
        else:
            #devs = ['/gpu:0']
            devs = ['/cpu:0']
        with tf.device(devs[0]):
            # train the 3D ConvNet
            for seq in range(1,2):
                print '**************************************'
                print 'current sequence is ',seq
                print '**************************************'
                sess.run(tf.global_variables_initializer())
                ut_set1.splitTrainingTesting(seq)
                test_x,test_y = ut_set1.loadTesting_new()
                for i in range(1001):
                    train_x,train_y = ut_set1.loadTraining(15)
                    if i%100 == 0:
                        train_accuracy = accuracy.eval(feed_dict={x:train_x, y_: train_y, keep_prob: 1})
                        testF = np.mean([features.eval(feed_dict={x:xT,keep_prob: 1}) for xT in test_x],0)
                        test_accuracy = accuracyT.eval(feed_dict={featuresT:testF,y_:test_y})
                        print("step %d, training accuracy %g and testing accuracy %g"%(i, train_accuracy, test_accuracy))
                    train_step.run(feed_dict={x:train_x, y_: train_y, keep_prob:0.5})
                print '   '
                # test the trained network performance
                #saver.save(sess,'./model_0117.ckpt')
                #saver.restore(sess,'./model.ckpt')
if __name__ == "__main__":
    tf.app.run(main=main)