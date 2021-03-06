'''Defines the modle of the feature descriptors and classifiers'''
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1,'../common')
import common

'''weights initialization'''
def weight_variable(shape):
    #return tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer(stddev=0.1,seed=1024))
    return tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer())

'''biases initialization'''
def bias_variable(shape):
    return tf.get_variable('bias',shape,initializer=tf.constant_initializer(0.1))

'''3D convolution'''
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1,1], padding='SAME')

'''3D max pooling with strides 1,2,2 (d,h,w)'''
def max_pool3d_1x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                            strides=[1, 1, 2, 2, 1], padding='SAME')

'''3D max pooling with strides 2,2,2 (d,h,w)'''
def max_pool3d_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')

'''3D max pooling with strides 2,1,1 (d,h,w)'''
def max_pool3d_2x1x1(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 1, 1, 1],
                            strides=[1, 2, 1, 1, 1], padding='SAME')

'''3D max pooling with strides 4,2,2 (d,h,w)'''
def max_pool3d_4x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 4, 2, 2, 1],
                            strides=[1, 4, 2, 2, 1], padding='SAME')

'''Defines the model of the feature descriptor'''
class FeatureDescriptor:
    '''defines a feature descriptor based on a basic 3D ConvNet'''
    @staticmethod
    def c3d(x,frmSize,drop_var, nof_conv1 = 64, nof_conv2 = 128, nof_conv3 = 256, nof_conv4 = 256, noo_fc6 = 4096, noo_fc7 = 4096):
        if (drop_var != 1):
            is_training = True
        else:
            if_traning = False
        bn_en = True 
        # define the first convlutional layer
        with tf.variable_scope('conv1'):
            numOfFilters_conv1 = nof_conv1 
            W_conv1 = weight_variable([3,3,3,frmSize[2],numOfFilters_conv1])
            b_conv1 = bias_variable([numOfFilters_conv1])
            h_conv1 = conv3d(x, W_conv1) + b_conv1
            h_relu1 = tf.nn.relu(h_conv1)
            h_pool1 = max_pool3d_1x2x2(h_relu1)
            if bn_en:
                print('Enable BN')
                h_bn1 = tf.contrib.layers.batch_norm(h_pool1,is_training=is_training)
            else:
                print('Diable BN')
                h_bn1 = h_pool1

        # define the second convlutional layer
        with tf.variable_scope('conv2'):
            numOfFilters_conv2 = nof_conv2 
            W_conv2 = weight_variable([3,3,3,numOfFilters_conv1,numOfFilters_conv2])
            b_conv2 = bias_variable([numOfFilters_conv2])
            h_conv2 = conv3d(h_bn1, W_conv2) + b_conv2
            h_relu2 = tf.nn.relu(h_conv2)
            h_pool2 = max_pool3d_2x2x2(h_relu2)
            if bn_en:
                h_bn2 = tf.contrib.layers.batch_norm(h_pool2,is_training=is_training)
            else:
                h_bn2 = h_pool2
    
        # define the 3rd convlutional layer
        with tf.variable_scope('conv3a'):
            numOfFilters_conv3 = nof_conv3 
            W_conv3a = weight_variable([3,3,3,numOfFilters_conv2,numOfFilters_conv3])
            b_conv3a = bias_variable([numOfFilters_conv3])
            h_conv3a = conv3d(h_bn2, W_conv3a) + b_conv3a
            h_relu3a = tf.nn.relu(h_conv3a)
            
        #with tf.variable_scope('conv3b'):
        #    numOfFilters_conv3b = nof_conv3 
        #    W_conv3b = weight_variable([3,3,3,numOfFilters_conv3a,numOfFilters_conv3b])
        #    b_conv3b = bias_variable([numOfFilters_conv3b])
        #    h_conv3b = tf.nn.relu(conv3d(h_conv3a, W_conv3b) + b_conv3b)
            h_pool3 = max_pool3d_2x2x2(h_relu3a)    
    
        # define the 4rd convlutional layer
        with tf.variable_scope('conv4a'):
            numOfFilters_conv4 = nof_conv4
            W_conv4a = weight_variable([3,3,3,numOfFilters_conv3,numOfFilters_conv4])
            b_conv4a = bias_variable([numOfFilters_conv4])
            h_conv4a = conv3d(h_pool3, W_conv4a) + b_conv4a
            h_relu4a = tf.nn.relu(h_conv4a)
        #with tf.variable_scope('conv4b'):
        #    numOfFilters_conv4b = nof_conv4 
        #    W_conv4b = weight_variable([3,3,3,numOfFilters_conv4a,numOfFilters_conv4b])
        #    b_conv4b = bias_variable([numOfFilters_conv4b])
        #    h_conv4b = tf.nn.relu(conv3d(h_conv4a, W_conv4b) + b_conv4b)
    
        # define the 5rd convlutional layer
        conv5a_en = True 
        if conv5a_en == True:
            h_pool4 = max_pool3d_2x2x2(h_relu4a)    
            with tf.variable_scope('conv5a'):
                numOfFilters_conv5 = numOfFilters_conv4 
                W_conv5a = weight_variable([3,3,3,numOfFilters_conv4,numOfFilters_conv5])
                b_conv5a = bias_variable([numOfFilters_conv5])
                h_conv5a = tf.nn.relu(conv3d(h_pool4, W_conv5a) + b_conv5a)
            #with tf.variable_scope('conv5b'):
            #    W_conv5b = weight_variable([3,3,3,numOfFilters_conv5,numOfFilters_conv5])
            #    b_conv5b = bias_variable([numOfFilters_conv5])
            #    h_conv5b = tf.nn.relu(conv3d(h_conv5a, W_conv5b) + b_conv5b)
                h_pool5 = max_pool3d_2x1x1(h_conv5a)    
        else:
            h_pool5 = max_pool3d_4x2x2(h_relu4a)    
    
        # define the full connected layer
        with tf.variable_scope('fc6'):
            numOfOutputs_fc6 = noo_fc6
            bn_fc6_en = False
            W_fc6 = weight_variable([int(frmSize[0]/16 * frmSize[1]/16) * numOfFilters_conv4, numOfOutputs_fc6])
            b_fc6 = bias_variable([numOfOutputs_fc6])
            h_pool4_flat = tf.reshape(h_pool5, [-1, int(frmSize[0]/16 * frmSize[1]/16) * numOfFilters_conv4])
            h_fc6 = tf.matmul(h_pool4_flat, W_fc6) + b_fc6 
            #h_fc6_bn = tf.contrib.layers.batch_norm(h_fc6,is_training=is_training)
            h_fc6_relu = tf.nn.relu(h_fc6)  
            h_fc6_drop = tf.nn.dropout(h_fc6_relu, drop_var) 
    
        # define the full connected layer fc7
        with tf.variable_scope('fc7'):
            numOfOutputs_fc7 = noo_fc7
            bn_fc7_en = False
            W_fc7 = weight_variable([numOfOutputs_fc6, numOfOutputs_fc7])
            b_fc7 = bias_variable([numOfOutputs_fc7])
            h_fc7 = tf.matmul(h_fc6_drop, W_fc7) + b_fc7
            #h_fc7_bn = tf.contrib.layers.batch_norm(h_fc7,is_training=is_training)
            h_fc7_relu = tf.nn.relu(h_fc7)
            h_fc7_drop = tf.nn.dropout(h_fc7_relu, drop_var)
        
        return h_fc7_drop

'''defines the models of the classifiers'''
class Classifier:
    '''defines a softmax classifier'''
    @staticmethod
    def softmax(features,numOfClasses):
        # softmax
        featuresDims = features.get_shape().as_list()[1]
        features_l2norm = tf.nn.l2_normalize(features,dim=1)
        with tf.variable_scope('sm'):
            W_sm = weight_variable([featuresDims, numOfClasses])
            b_sm = bias_variable([numOfClasses])
            #y_conv = tf.matmul(features_l2norm, W_sm) + b_sm 
            y_conv = tf.matmul(features, W_sm) + b_sm 
        return y_conv

'''define a softmax classifier class'''
class Softmax:
    def __init__(self,features,numOfClasses):
        # softmax
        featuresDims = features.get_shape().as_list()[1]
        features_l2norm = tf.nn.l2_normalize(features,dim=1)
        with tf.variable_scope('sm'):
            self.W_sm = weight_variable([featuresDims, numOfClasses])
            self.b_sm = bias_variable([numOfClasses])
            #y_conv = tf.matmul(features_l2norm, W_sm) + b_sm 
            self.y_conv = tf.matmul(features, self.W_sm) + self.b_sm 