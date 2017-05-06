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
    def __init__(self, numOfClasses,frmSize,nof_conv1 = 64, nof_conv2 = 128, nof_conv3 = 256, nof_conv4 = 256, noo_fc6 = 4096, noo_fc7 = 4096):
        # build the 3D ConvNet
        # define the input and output variables
        self._x = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._featuresT = tf.placeholder(tf.float32,(None,noo_fc7))
        self._keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        
        with tf.variable_scope('feature_descriptor_g') as scope:
            self._features = model.FeatureDescriptor.c3d(self._x,frmSize,self._keep_prob,nof_conv1, nof_conv2, nof_conv3, nof_conv4, noo_fc6, noo_fc7)
        with tf.variable_scope('classifier') as scope:
            self._classifier = model.Softmax(self._features,numOfClasses)
            self._y_conv = self._classifier.y_conv
            scope.reuse_variables()
            self._classifierT = model.Softmax(self._featuresT,numOfClasses)
            self._y_convT = self._classifierT.y_conv
            self._y_sm = tf.nn.softmax(self._y_convT)
            
            
        with tf.device(common.Vars.dev[-1]):
            # Train and evaluate the model
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            #self._train_step = tf.train.AdamOptimizer(learning_rate=0.05, epsilon=0.01).minimize(cross_entropy,var_list=self.getClassifierVars())
            #self._train_step = tf.train.AdamOptimizer(learning_rate=0.005, epsilon=0.01).minimize(cross_entropy)
            self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=0.01).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self._y_conv,1), tf.argmax(self._y_,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            correct_predictionT = tf.equal(tf.argmax(self._y_convT,1), tf.argmax(self._y_,1))
            self._accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
        return None
    
    def getFeature(self,test_x,sess):
        return self._features.eval(feed_dict={self._x:test_x,self._keep_prob:1},session=sess)
    
    def getClassifierVars(self):
        return([self._classifier.W_sm,self._classifier.b_sm])
    
    def train(self, train_x,train_y,sess, learning_rate = 0.005):
        with sess.as_default():
            #self._train_step.run(feed_dict={self._x:train_x, self._y_:train_y, self._keep_prob:0.5})
            self._train_step.run(feed_dict={self._lr: learning_rate, self._x:train_x, self._y_:train_y, self._keep_prob:0.5})
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
    
    def evaluateProb(self, test_x, sess):
        with sess.as_default():
            testF = np.mean([self._features.eval(feed_dict={self._x:xT,self._keep_prob: 1}) for xT in test_x],0)
            test_prob = self._y_convT.eval(feed_dict={self._featuresT:testF})
        return test_prob
    
    def obs(self,test_x,test_y,sess):
        with sess.as_default():
            prob = self.evaluateProb(test_x, sess)
            top2y = np.array[np.argsort(prob)[:,-1],np.argsort(prob)[:,-2]]
            top2y = top2y.transpose(1,0) 
            print(top2y,' vs ',test_y)
        return None
            
    

class C3DNET_2F1C:
    def __init__(self, numOfClasses,frmSize, shareFeatureVariable = True):
        # build the 3D ConvNet
        # define the input and output variables
        self._x0 = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._x1 = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._featuresT = tf.placeholder(tf.float32,(None,2048*2))
        self._keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        
        if shareFeatureVariable == True:
            with tf.variable_scope('feature_descriptor_a0') as scope:
                self._features0 = model.FeatureDescriptor.c3d(self._x0,frmSize,self._keep_prob)
                scope.reuse_variables()
                self._features1 = model.FeatureDescriptor.c3d(self._x1,frmSize,self._keep_prob)
        else:
            with tf.variable_scope('feature_descriptor_a0') as scope:
                self._features0 = model.FeatureDescriptor.c3d(self._x0,frmSize,self._keep_prob)
            with tf.variable_scope('feature_descriptor_a1') as scope:
                self._features1 = model.FeatureDescriptor.c3d(self._x1,frmSize,self._keep_prob)
            
        with tf.variable_scope('classifier_2f1c') as scope:
            features = tf.concat([self._features0, self._features1],1)
            self._classifier = model.Softmax(features,numOfClasses)
            self._y_conv = self._classifier.y_conv
            scope.reuse_variables()
            self._classifierT = model.Softmax(self._featuresT,numOfClasses)
            self._y_convT = self._classifierT.y_conv
        
        with tf.device(common.Vars.dev[-1]):
            # Train and evaluate the model
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            #self._train_step = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.01).minimize(cross_entropy)
            self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=0.01).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self._y_conv,1), tf.argmax(self._y_,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            correct_predictionT = tf.equal(tf.argmax(self._y_convT,1), tf.argmax(self._y_,1))
            self._accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
    
    def train(self, train_x0, train_x1, train_y,sess,learning_rate = 1e-4):
        with sess.as_default():
            self._train_step.run(feed_dict={self._lr: learning_rate, self._x0:train_x0, self._x1:train_x1, self._y_:train_y, self._keep_prob:0.5})
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
    
    def evaluateProb(self, test_x0, test_x1, sess):
        with sess.as_default():
            testF0 = np.mean([self._features0.eval(feed_dict={self._x0:xT,self._keep_prob: 1}) for xT in test_x0],0)
            testF1 = np.mean([self._features1.eval(feed_dict={self._x1:xT,self._keep_prob: 1}) for xT in test_x1],0)
            testF = np.concatenate((testF0, testF1),1)
            test_prob = self._y_convT.eval(feed_dict={self._featuresT:testF})
        return test_prob

class C3DNET_3F1C:
    def __init__(self, numOfClasses,frmSize):
        # build the 3D ConvNet
        # define the input and output variables
        self._x  = tf.placeholder(tf.float32, (None,16) + frmSize[0])
        self._x0 = tf.placeholder(tf.float32, (None,16) + frmSize[1])
        self._x1 = tf.placeholder(tf.float32, (None,16) + frmSize[1])
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._featuresT = tf.placeholder(tf.float32,(None,2048*3))
        self._keep_prob = tf.placeholder(tf.float32)
        
        with tf.variable_scope('feature_descriptor_g') as scope:
            self._features_g  = model.FeatureDescriptor.c3d(self._x,frmSize[0],self._keep_prob)
        with tf.variable_scope('feature_descriptor_a0') as scope:
            self._features_a0 = model.FeatureDescriptor.c3d(self._x0,frmSize[1],self._keep_prob)
        with tf.variable_scope('feature_descriptor_a1') as scope:
            self._features_a1 = model.FeatureDescriptor.c3d(self._x1,frmSize[1],self._keep_prob)
            
        with tf.variable_scope('classifier_3f1c') as scope:
            features = tf.concat([self._features_g, self._features_a0, self._features_a1],1)
            self._classifier = model.Softmax(features,numOfClasses)
            self._y_conv = self._classifier.y_conv
            scope.reuse_variables()
            self._classifierT = model.Softmax(self._featuresT,numOfClasses)
            self._y_convT = self._classifierT.y_conv 
        
        with tf.device(common.Vars.dev[0]):
            # Train and evaluate the model
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            #self._train_step = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=0.01).minimize(cross_entropy,var_list=self.getClassifierVars())
            self._train_step = tf.train.AdamOptimizer(learning_rate=0.003, epsilon=0.00001).minimize(cross_entropy,var_list=self.getClassifierVars())
            correct_prediction = tf.equal(tf.argmax(self._y_conv,1), tf.argmax(self._y_,1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            correct_predictionT = tf.equal(tf.argmax(self._y_convT,1), tf.argmax(self._y_,1))
            self._accuracyT = tf.reduce_mean(tf.cast(correct_predictionT, tf.float32))
    
    def getClassifierVars(self):
        return([self._classifier.W_sm,self._classifier.b_sm])
    
    def train(self, train_x, train_x0, train_x1, train_y,sess):
        with sess.as_default():
            self._train_step.run(feed_dict={self._x:train_x, self._x0:train_x0, self._x1:train_x1, self._y_:train_y, self._keep_prob:0.5})
        return None
    
    def evaluate(self, test_x, test_x0, test_x1, test_y, sess):
        with sess.as_default():
            if test_x0.ndim == 6:
                testFg = np.mean([self._features_g.eval(feed_dict={self._x:xT,self._keep_prob: 1}) for xT in test_x],0)
                testF0 = np.mean([self._features_a0.eval(feed_dict={self._x0:xT,self._keep_prob: 1}) for xT in test_x0],0)
                testF1 = np.mean([self._features_a1.eval(feed_dict={self._x1:xT,self._keep_prob: 1}) for xT in test_x1],0)
                testF = np.concatenate((testFg, testF0, testF1),1)
                test_accuracy = self._accuracyT.eval(feed_dict={self._featuresT:testF, self._y_:test_y})
            else:
                test_accuracy = self._accuracy.eval(feed_dict={self._x: test_x, self._x0:test_x0, self._x1:test_x1, self._y_:test_y, self._keep_prob:1})
        return test_accuracy 
    
    def test(self, test_x, test_x0, test_x1, test_y, sess):
        if test_x0.ndim == 6:
            test_accuracy = np.mean([self.evaluate(np.reshape(x, common.tupleInsert(x.shape,1,1)), \
                                                   np.reshape(x0,common.tupleInsert(x0.shape,1,1)), \
                                                   np.reshape(x1,common.tupleInsert(x1.shape,1,1)), \
                                                   np.reshape(y,(1,)+y.shape),sess) \
                                                   for x,x0,x1,y in zip(test_x.transpose(1,0,2,3,4,5), test_x0.transpose(1,0,2,3,4,5),test_x1.transpose(1,0,2,3,4,5),test_y)])
        else:
            test_accuracy = np.mean([self.evaluate(np.reshape(x,(1,) +x.shape),
                                                   np.reshape(x0,(1,)+x0.shape), \
                                                   np.reshape(x1,(1,)+x1.shape), \
                                                   np.reshape(y,(1,)+y.shape),sess) \
                                                   for x,x0,x1,y in zip(test_x,test_x0,test_x1,test_y)])
        return test_accuracy 