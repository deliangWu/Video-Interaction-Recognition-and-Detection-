
import numpy as np
import os
import tensorflow as tf
from os.path import join 
import sys
import model_vf
sys.path.insert(1,'../datasets')
sys.path.insert(1,'../common')
sys.path.insert(1,'../train')
import ut_interaction as ut
import common
import videoPreProcess as vpp
import network


class C3DNET:
    def __init__(self, frmSize,visual_layer, nof_conv1 = 64, nof_conv2 = 128, nof_conv3 = 256, nof_conv4 = 256, noo_fc6 = 4096, noo_fc7 = 4096):
        # build the 3D ConvNet
        # define the input and output variables
        self._nof_conv1 = nof_conv1
        self._nof_conv2 = nof_conv2
        self._nof_conv3 = nof_conv3
        self._nof_conv4 = nof_conv4
        self._x = tf.placeholder(tf.float32, (None,16) + frmSize)
        if visual_layer == 1:
            self._features_ph = tf.placeholder(tf.float32,(None,16,frmSize[0]/2,frmSize[1]/2,nof_conv1))
        elif visual_layer == 2:
            self._features_ph = tf.placeholder(tf.float32,(None,8, frmSize[0]/4,frmSize[1]/4,nof_conv2))
        elif visual_layer == 3:
            self._features_ph = tf.placeholder(tf.float32,(None,4, frmSize[0]/8,frmSize[1]/8,nof_conv3))
        else:
            self._features_ph = tf.placeholder(tf.float32,(None,1, frmSize[0]/16,frmSize[1]/16,nof_conv4))
            
        self._keep_prob = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        
        with tf.variable_scope('feature_descriptor_g') as scope:
            self._features = model_vf.FeatureDescriptor.c3d(self._x,frmSize,self._keep_prob,nof_conv1, nof_conv2, nof_conv3, nof_conv4, noo_fc6, noo_fc7,rl=visual_layer)
            scope.reuse_variables()
            self._unConv = model_vf.FeatureDescriptor.c3d_v(self._features_ph,frmSize, nof_conv1, nof_conv2, nof_conv3, nof_conv4,layer=visual_layer)
        return None
    
    def getFeature(self,test_x,sess):
        return self._features.eval(feed_dict={self._x:test_x,self._keep_prob:1}, session=sess)
    
    def visualize(self,test_x,sess):
        with sess.as_default():
            features_gen = self._features.eval(feed_dict={self._x:test_x,self._keep_prob:1})
            videoOuts = []
            for i in range(64):
                features = features_gen.copy()
                features[:,:,:,:,:i] = 0
                features[:,:,:,:,i+1:] = 0
                print(features.shape)
                videoOut = self._unConv.eval(feed_dict={self._features_ph:features})
                videoOuts.append(videoOut)
        return np.array(videoOuts)

def main(argv):
    # ***********************************************************
    # define the network
    # ***********************************************************
    frmSize = (112,128,3)
    with tf.variable_scope('top') as scope:
        c3d = C3DNET(frmSize, visual_layer=1, nof_conv1=32, nof_conv2=128, nof_conv3=256, nof_conv4=512, noo_fc6=4096, noo_fc7=4096)
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
    ut_set = ut.ut_interaction_set1(frmSize,numOfClasses=6)
    
    # ***********************************************************
    # Train and test the network
    # ***********************************************************
    seqRange = range(1,2)
    for seq in seqRange:
        with sess.as_default():
            sess.run(initVars)
        ut_set.splitTrainingTesting(seq, loadTrainingEn=False)
        ut_set.loadTrainingAll()
        train_x,train_y = ut_set.loadTrainingBatch(10)
        test_x,test_lable = ut_set.loadTesting()
        #videoIn = test_x[2][0]
        videoIn = train_x[2]
        vpp.videoPlay(videoIn,fps=10)
            
        # load trained network  
        saver_feature_g = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.feature_g_VarsList])
        #saver_classifier = tf.train.Saver([tf.get_default_graph().get_tensor_by_name(varName) for varName in common.Vars.classifier_sm_VarsList])
        saver_feature_g.restore(sess,join(common.path.variablePath, 'c3d_train_on_ut_set1_' + str(seq) + '_fg.ckpt'))
        #saver_classifier.restore(sess,join(common.path.variablePath, 'c3d_train_on_ut_set1_' + str(seq) + '_c7.ckpt'))
        
        videoIn = np.reshape(videoIn,(1,)+videoIn.shape)
        visualFeatures = c3d.visualize(videoIn, sess)    
        i = 0
        for visualFeature in visualFeatures:
            vf = vpp.videoNorm(visualFeature[0])
            videoShow = np.concatenate([videoIn[0],vf],1)
            vpp.videoPlay(videoShow,fps=1)
            vpp.videoSave(videoShow,'feature_'+str(i)+'.avi')
            i+=1
 
                
if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)