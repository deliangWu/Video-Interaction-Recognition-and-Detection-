import numpy as np
import os
import tensorflow as tf
import sys
sys.path.insert(1,'../model')
sys.path.insert(1,'../datasets')
import model
import ut_interaction as ut


class C3DNET:
    def __init__(self, numOfClasses,frmSize):
        # build the 3D ConvNet
        # define the input and output variables
        self._x = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._featuresT = tf.placeholder(tf.float32,(None,4096))
        self._keep_prob = tf.placeholder(tf.float32)
        
        with tf.variable_scope('feature_descriptor') as scope:
            self._features = model.FeatureDescriptor.c3d(self._x,frmSize,self._keep_prob)
        with tf.variable_scope('classifier') as scope:
            self._y_conv = model.Classifier.softmax(self._features,numOfClasses)
            scope.reuse_variables()
            self._y_convT = model.Classifier.softmax(self._featuresT,numOfClasses)
        
        # Train and evaluate the model
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
        self._train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self._y_conv,1), tf.argmax(self._y_,1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        correct_predictionT = tf.equal(tf.argmax(self._y_convT,1), tf.argmax(self._y_,1))
        self._accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
        if os.name == 'nt':
            self._devs = ['/cpu:0']
        else:
            self._devs = ['/gpu:0']
    
    def train(self, train_x,train_y,sess):
        with sess.as_default():
            self._train_step.run(feed_dict={self._x:train_x, self._y_: train_y, self._keep_prob:0.5})
        return None
    
    def evaluate(self, test_x,test_y,sess):
        with sess.as_default():
            if test_x.ndim == 6:
                testF = np.mean([self._features.eval(feed_dict={self._x:xT,self._keep_prob: 1}) for xT in test_x],0)
                test_accuracy = self._accuracyT.eval(feed_dict={self._featuresT:testF, self._y_:test_y})
            else:
                test_accuracy = self._accuracy.eval(feed_dict={self._x:test_x, self._y_: test_y, self._keep_prob: 1})
        return test_accuracy 
    
