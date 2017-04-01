import numpy as np
import os
import tensorflow as tf
import sys
sys.path.insert(1,'../model')
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../common')
import model
import ut_interaction as ut
import common


class C3DNET:
    def __init__(self, numOfClasses,frmSize,nof_conv1 = 32, nof_conv2 = 64, nof_conv3 = 128):
        # build the 3D ConvNet
        # define the input and output variables
        self._x = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._featuresT = tf.placeholder(tf.float32,(None,4096))
        self._keep_prob = tf.placeholder(tf.float32)
        
        with tf.variable_scope('feature_descriptor') as scope:
            self._features = model.FeatureDescriptor.c3d(self._x,frmSize,self._keep_prob,nof_conv1, nof_conv2, nof_conv3)
        with tf.variable_scope('classifier') as scope:
            self._y_conv = model.Classifier.softmax(self._features,numOfClasses)
            scope.reuse_variables()
            self._y_convT = model.Classifier.softmax(self._featuresT,numOfClasses)
        
        with tf.device(common.Vars.dev[-1]):
            # Train and evaluate the model
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            self._train_step = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=0.01).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self._y_conv,1), tf.argmax(self._y_,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            correct_predictionT = tf.equal(tf.argmax(self._y_convT,1), tf.argmax(self._y_,1))
            self._accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
    
    def train(self, train_x,train_y,sess):
        with sess.as_default():
            self._train_step.run(feed_dict={self._x:train_x, self._y_: train_y, self._keep_prob:0.5})
        return None
    
    def evaluate(self, test_x, test_y, sess):
        with sess.as_default():
            if test_x.ndim == 6:
                testF = np.mean([self._features.eval(feed_dict={self._x:xT,self._keep_prob: 1}) for xT in test_x],0)
                test_accuracy = self._accuracyT.eval(feed_dict={self._featuresT:testF, self._y_:test_y})
            else:
                test_accuracy = self._accuracy.eval(feed_dict={self._x:test_x, self._y_:test_y, self._keep_prob: 1})
        return test_accuracy 
    
    def test(self, test_x, test_y, sess):
        if test_x.ndim == 6:
            test_accuracy = np.mean([self.evaluate(np.reshape(x,common.tupleInsert(x.shape,1,1)),np.reshape(y,(1,)+y.shape),sess) for x,y in zip(test_x.transpose(1,0,2,3,4,5),test_y)])
        else:
            test_accuracy = np.mean([self.evaluate(np.reshape(x,(1,)+x.shape),np.reshape(y,(1,)+y.shape),sess) for x,y in zip(test_x,test_y)])
        return test_accuracy 
    

class C3DNET_2F1C:
    def __init__(self, numOfClasses,frmSize, shareFeatureVariable = True):
        # build the 3D ConvNet
        # define the input and output variables
        self._x0 = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._x1 = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._featuresT = tf.placeholder(tf.float32,(None,8192))
        self._keep_prob = tf.placeholder(tf.float32)
        
        if shareFeatureVariable == True:
            with tf.variable_scope('feature_descriptor') as scope:
                self._features0 = model.FeatureDescriptor.c3d(self._x0,frmSize,self._keep_prob)
                scope.reuse_variables()
                self._features1 = model.FeatureDescriptor.c3d(self._x1,frmSize,self._keep_prob)
        else:
            with tf.variable_scope('feature_descriptor0') as scope:
                self._features0 = model.FeatureDescriptor.c3d(self._x0,frmSize,self._keep_prob)
            with tf.variable_scope('feature_descriptor1') as scope:
                self._features1 = model.FeatureDescriptor.c3d(self._x1,frmSize,self._keep_prob)
            
        with tf.variable_scope('classifier_2f1c') as scope:
            features = tf.concat([self._features0, self._features1],1)
            self._y_conv = model.Classifier.softmax(features, numOfClasses)
            scope.reuse_variables()
            self._y_convT = model.Classifier.softmax(self._featuresT,numOfClasses)
        
        with tf.device(common.Vars.dev[-1]):
            # Train and evaluate the model
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            self._train_step = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=0.01).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self._y_conv,1), tf.argmax(self._y_,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            correct_predictionT = tf.equal(tf.argmax(self._y_convT,1), tf.argmax(self._y_,1))
            self._accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
    
    def train(self, train_x0, train_x1, train_y,sess):
        with sess.as_default():
            self._train_step.run(feed_dict={self._x0:train_x0, self._x1:train_x1, self._y_:train_y, self._keep_prob:0.5})
        return None
    
    def evaluate(self, test_x0, test_x1, test_y, sess):
        with sess.as_default():
            if test_x0.ndim == 6:
                testF0 = np.mean([self._features0.eval(feed_dict={self._x0:xT,self._keep_prob: 1}) for xT in test_x0],0)
                testF1 = np.mean([self._features1.eval(feed_dict={self._x1:xT,self._keep_prob: 1}) for xT in test_x1],0)
                testF = np.concatenate((testF0, testF1),1)
                test_accuracy = self._accuracyT.eval(feed_dict={self._featuresT:testF, self._y_:test_y})
            else:
                test_accuracy = self._accuracy.eval(feed_dict={self._x0:test_x0, self._x1:test_x1, self._y_:test_y, self._keep_prob:1})
        return test_accuracy 
    
    def test(self, test_x0, test_x1, test_y, sess):
        if test_x0.ndim == 6:
            test_accuracy = np.mean([self.evaluate(np.reshape(x0,common.tupleInsert(x0.shape,1,1)), \
                                                   np.reshape(x1,common.tupleInsert(x1.shape,1,1)), \
                                                   np.reshape(y,(1,)+y.shape),sess) \
                                                   for x0,x1,y in zip(test_x0.transpose(1,0,2,3,4,5),test_x1.transpose(1,0,2,3,4,5),test_y)])
        else:
            test_accuracy = np.mean([self.evaluate(np.reshape(x0,(1,)+x0.shape),
                                                   np.reshape(x1,(1,)+x1.shape), \
                                                   np.reshape(y,(1,)+y.shape),sess) \
                                                   for x0,x1,y in zip(test_x0,test_x1,test_y)])
        return test_accuracy 
