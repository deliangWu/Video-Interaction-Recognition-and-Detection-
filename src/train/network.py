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
        self._keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
       
        # define feature descriptor and classifier 
        with tf.device(common.Vars.dev[0]):
            with tf.variable_scope('feature_descriptor_g') as scope:
                self._features = model.FeatureDescriptor.c3d(self._x,frmSize,self._keep_prob,nof_conv1, nof_conv2, nof_conv3, nof_conv4, noo_fc6, noo_fc7)
            with tf.variable_scope('classifier') as scope:
                self._classifier = model.Softmax(self._features,numOfClasses)
                self._y_conv = self._classifier.y_conv
            
        # Train and evaluate the model
        with tf.device(common.Vars.dev[0]):
            self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            #self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=0.01).minimize(self._cross_entropy)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=0.01).minimize(self._cross_entropy)
        return None
    
    def getFeature(self,test_x,sess):
        return self._features.eval(feed_dict={self._x:test_x,self._keep_prob:1},session=sess)
    
    def getLoss(self,test_x,test_y,sess):
        if test_x.ndim == 6:
            return np.mean([self._cross_entropy.eval(feed_dict={self._x:x,self._y_:test_y,self._keep_prob:1},session=sess) for x in test_x])
        else:
            return self._cross_entropy.eval(feed_dict={self._x:test_x,self._y_:test_y,self._keep_prob:1},session=sess)
    
    def getClassifierVars(self):
        return([self._classifier.W_sm,self._classifier.b_sm])
    
    def train(self, train_x,train_y,sess, learning_rate = 0.005):
        with sess.as_default():
            #self._train_step.run(feed_dict={self._x:train_x, self._y_:train_y, self._keep_prob:0.5})
            self._train_step.run(feed_dict={self._lr: learning_rate, self._x:train_x, self._y_:train_y, self._keep_prob:0.5})
        return None
    
    def top2Accu(self, test_x,test_y,sess):
        with sess.as_default():
            y_conv = []
            if test_x.ndim == 6:
                for single_test_x in test_x.transpose(1,0,2,3,4,5):
                    y_conv.append(np.mean([self._y_conv.eval(feed_dict = {self._x:np.reshape(x,(1,)+x.shape), self._keep_prob:1})[0]/3 for x in single_test_x],0))
            else:
                for single_test_x in test_x:
                    y_conv.append(self._y_conv.eval(feed_dict = {self._x:np.reshape(single_test_x,(1,)+single_test_x.shape), self._keep_prob:1})[0])
            y_conv = np.array(y_conv)
            # top1 accuracy
            top1_accu = np.mean(np.equal(np.argmax(y_conv,1),np.argmax(test_y,1)))
            # top2 accuracy 
            top2y = np.array([np.argsort(y_conv)[:,-1],np.argsort(y_conv)[:,-2]]).transpose(1,0)
            obs = np.array([np.argsort(y_conv)[:,-1],np.argsort(y_conv)[:,-2],np.argmax(test_y,1)]).transpose(1,0) 
            #if test_x.ndim == 6 and top1_accu > 0.8:
                #print(obs)
            top2_accu = np.mean([int(np.argmax(y) in t2y) for y,t2y in zip(test_y,top2y)])
        return(top1_accu,top2_accu)
    
    def obs(self,test_x,test_y,sess):
        with sess.as_default():
            probs = np.array([self._y_conv.eval(feed_dict={self._x:x,self._keep_prob:1}) for x in test_x])
            top2y = np.array([[np.argsort(prob)[:,-1],np.argsort(prob)[:,-2]] for prob in probs])
            top2y = top2y.transpose(2,1,0) 
            print(top2y,' vs ',np.argmax(test_y,1))
        return None
    
    def evaluateProb(self,test_x,sess):
        with sess.as_default():
            y_conv = []
            if test_x.ndim == 6:
                for single_test_x in test_x.transpose(1,0,2,3,4,5):
                    y_conv.append(np.mean([self._y_conv.eval(feed_dict = {self._x:np.reshape(x,(1,)+x.shape), self._keep_prob:1})[0]/3 for x in single_test_x],0))
            else:
                for single_test_x in test_x:
                    y_conv.append(self._y_conv.eval(feed_dict = {self._x:np.reshape(single_test_x,(1,)+single_test_x.shape), self._keep_prob:1})[0])
            probs = np.array(y_conv)
            probs = tf.nn.softmax(probs)
        return probs
    
class C3DNET_2F1C:
    def __init__(self, numOfClasses,frmSize, nof_conv1,nof_conv2,nof_conv3,nof_conv4,noo_fc6,noo_fc7, shareFeatureVariable = True):
        # build the 3D ConvNet
        # define the input and output variables
        self._x0 = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._x1 = tf.placeholder(tf.float32, (None,16) + frmSize)
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        
        if shareFeatureVariable == True:
            with tf.variable_scope('feature_descriptor_a0') as scope:
                self._features0 = model.FeatureDescriptor.c3d(self._x0,frmSize,self._keep_prob,nof_conv1, nof_conv2, nof_conv3, nof_conv4, noo_fc6, noo_fc7)
                scope.reuse_variables()
                self._features1 = model.FeatureDescriptor.c3d(self._x1,frmSize,self._keep_prob,nof_conv1, nof_conv2, nof_conv3, nof_conv4, noo_fc6, noo_fc7)
        else:
            with tf.variable_scope('feature_descriptor_a0') as scope:
                self._features0 = model.FeatureDescriptor.c3d(self._x0,frmSize,self._keep_prob)
            with tf.variable_scope('feature_descriptor_a1') as scope:
                self._features1 = model.FeatureDescriptor.c3d(self._x1,frmSize,self._keep_prob)
            
        with tf.variable_scope('classifier_2f1c') as scope:
            features = tf.concat([self._features0, self._features1],1)
            self._classifier = model.Softmax(features,numOfClasses)
            self._y_conv = self._classifier.y_conv
        
        with tf.device(common.Vars.dev[-1]):
            # Train and evaluate the model
            self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            #self._train_step = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.01).minimize(cross_entropy)
            self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=0.01).minimize(self._cross_entropy)
    
    
    def train(self, train_x0, train_x1, train_y,sess,learning_rate = 1e-4):
        with sess.as_default():
            self._train_step.run(feed_dict={self._lr: learning_rate, self._x0:train_x0, self._x1:train_x1, self._y_:train_y, self._keep_prob:0.5})
        return None
    
    def getLoss(self,test_x0, test_x1,test_y,sess):
        if test_x0.ndim == 6:
            return self._cross_entropy.eval(feed_dict={self._x0:test_x0[1], self._x1:test_x1[1], self._y_:test_y,self._keep_prob:1},session=sess)
        else:
            return self._cross_entropy.eval(feed_dict={self._x0:test_x0, self._x1:test_x1, self._y_:test_y,self._keep_prob:1},session=sess)
            
    
    def top2Accu(self, test_x0, test_x1,test_y,sess):
        with sess.as_default():
            y_conv = []
            if test_x0.ndim == 6:
                for single_test_x0,single_test_x1 in zip(test_x0.transpose(1,0,2,3,4,5),test_x1.transpose(1,0,2,3,4,5)):
                    y_conv.append(np.mean([self._y_conv.eval(feed_dict = {self._x0:np.reshape(x0, (1,)+x0.shape), \
                                                                          self._x1:np.reshape(x1, (1,)+x1.shape), \
                                                                          self._keep_prob:1})[0]/3 \
                                           for x0,x1 in zip(single_test_x0,single_test_x1)],0))
            else:
                for single_test_x0,single_test_x1 in zip(test_x0,test_x1):
                    y_conv.append(self._y_conv.eval(feed_dict = {self._x0:np.reshape(single_test_x0,(1,)+single_test_x0.shape), \
                                                                 self._x1:np.reshape(single_test_x1,(1,)+single_test_x1.shape), \
                                                                 self._keep_prob:1})[0])
            y_conv = np.array(y_conv)
            # top1 accuracy
            top1_accu = np.mean(np.equal(np.argmax(y_conv,1),np.argmax(test_y,1)))
            # top2 accuracy 
            top2y = np.array([np.argsort(y_conv)[:,-1],np.argsort(y_conv)[:,-2]]).transpose(1,0)
            top2_accu = np.mean([int(np.argmax(y) in t2y) for y,t2y in zip(test_y,top2y)])
        return(top1_accu,top2_accu)
    

class C3DNET_3F1C:
    def __init__(self, numOfClasses,frmSize, nof_conv1,nof_conv2,nof_conv3,nof_conv4,noo_fc6,noo_fc7):
        # build the 3D ConvNet
        # define the input and output variables
        self._x = tf.placeholder(tf.float32, (None,16) + frmSize[0])
        self._x0 = tf.placeholder(tf.float32, (None,16) + frmSize[1])
        self._x1 = tf.placeholder(tf.float32, (None,16) + frmSize[1])
        self._y_ = tf.placeholder(tf.float32, (None, numOfClasses))
        self._keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        with tf.device(common.Vars.dev[0]):
            with tf.variable_scope('feature_descriptor_g') as scope:
                self._features_g = model.FeatureDescriptor.c3d(self._x,frmSize[0],self._keep_prob)
        with tf.device(common.Vars.dev[1]):
            with tf.variable_scope('feature_descriptor_a0') as scope:
                self._features_a0 = model.FeatureDescriptor.c3d(self._x0,frmSize[1],self._keep_prob)
            with tf.variable_scope('feature_descriptor_a1') as scope:
                self._features_a1 = model.FeatureDescriptor.c3d(self._x1,frmSize[1],self._keep_prob)
            
        with tf.device(common.Vars.dev[1]):
            with tf.variable_scope('classifier_3f1c') as scope:
                features = tf.concat([self._features_g, self._features_a0, self._features_a1],1)
                self._classifier = model.Softmax(features,numOfClasses)
                self._y_conv = self._classifier.y_conv
        
        with tf.device(common.Vars.dev[-1]):
            # Train and evaluate the model
            self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._y_conv, labels=self._y_))
            #self._train_step = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.01).minimize(cross_entropy)
            self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=0.01).minimize(self._cross_entropy)
    
    
    def train(self, train_x, train_x0, train_x1, train_y,sess,learning_rate = 1e-4):
        with sess.as_default():
            self._train_step.run(feed_dict={self._lr: learning_rate, self._x: train_x,
                                                                     self._x0:train_x0,
                                                                     self._x1:train_x1,
                                                                     self._y_:train_y, self._keep_prob:0.5})
        return None
    
    def getLoss(self,test_x, test_x0, test_x1,test_y,sess):
        if test_x0.ndim == 6:
            return self._cross_entropy.eval(feed_dict={self._x:test_x[0], self._x0:test_x0[0], self._x1:test_x1[0], self._y_:test_y,self._keep_prob:1},session=sess)
        else:
            return self._cross_entropy.eval(feed_dict={self._x:test_x,    self._x0:test_x0,    self._x1:test_x1,    self._y_:test_y,self._keep_prob:1},session=sess)
            
    
    def top2Accu(self, test_x, test_x0, test_x1,test_y,sess):
        with sess.as_default():
            y_conv = []
            if test_x.ndim == 6:
                for single_test_x, single_test_x0,single_test_x1 in zip(test_x.transpose(1,0,2,3,4,5), test_x0.transpose(1,0,2,3,4,5),test_x1.transpose(1,0,2,3,4,5)):
                    y_conv.append(np.mean([self._y_conv.eval(feed_dict = {self._x: np.reshape(x, (1,)+x.shape),
                                                                          self._x0:np.reshape(x0, (1,)+x0.shape),
                                                                          self._x1:np.reshape(x1, (1,)+x1.shape),
                                                                          self._keep_prob:1})[0]/3 \
                                           for x, x0,x1 in zip(single_test_x, single_test_x0,single_test_x1)],0))
            else:
                for single_test_x, single_test_x0,single_test_x1 in zip(test_x, test_x0,test_x1):
                    y_conv.append(self._y_conv.eval(feed_dict = {self._x: np.reshape(single_test_x, (1,)+single_test_x.shape),
                                                                 self._x0:np.reshape(single_test_x0,(1,)+single_test_x0.shape),
                                                                 self._x1:np.reshape(single_test_x1,(1,)+single_test_x1.shape),
                                                                 self._keep_prob:1})[0])
            y_conv = np.array(y_conv)
            # top1 accuracy
            top1_accu = np.mean(np.equal(np.argmax(y_conv,1),np.argmax(test_y,1)))
            # top2 accuracy 
            top2y = np.array([np.argsort(y_conv)[:,-1],np.argsort(y_conv)[:,-2]]).transpose(1,0)
            top2_accu = np.mean([int(np.argmax(y) in t2y) for y,t2y in zip(test_y,top2y)])
        return(top1_accu,top2_accu)
