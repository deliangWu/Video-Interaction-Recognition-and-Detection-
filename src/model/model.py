import numpy as np
import tensorflow as tf

def weight_variable(shape):
    return tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer(stddev=1.1))

def bias_variable(shape):
    return tf.get_variable('bias',shape,initializer=tf.truncated_normal_initializer(stddev=1.1))

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1,1], padding='SAME')

def max_pool3d_1x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                            strides=[1, 1, 2, 2, 1], padding='SAME')

def max_pool3d_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')

def max_pool3d_2x1x1(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 1, 1, 1],
                            strides=[1, 2, 1, 1, 1], padding='SAME')

def max_pool3d_4x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 4, 2, 2, 1],
                            strides=[1, 4, 2, 2, 1], padding='SAME')

class FeatureDescriptor:
    @staticmethod
    def c3d(x,frmSize,drop_var):
        # define the first convlutional layer
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([3,3,3,frmSize[2],64])
            b_conv1 = bias_variable([64])
            h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1)
            h_pool1 = max_pool3d_1x2x2(h_conv1)    
    
        # define the second convlutional layer
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([3,3,3,64,128])
            b_conv2 = bias_variable([128])
            h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool3d_2x2x2(h_conv2)    
    
        # define the 3rd convlutional layer
        with tf.variable_scope('conv3a'):
            W_conv3a = weight_variable([3,3,3,128,256])
            b_conv3a = bias_variable([256])
            h_conv3a = tf.nn.relu(conv3d(h_pool2, W_conv3a) + b_conv3a)
        with tf.variable_scope('conv3b'):
            W_conv3b = weight_variable([3,3,3,256,256])
            b_conv3b = bias_variable([256])
            h_conv3b = tf.nn.relu(conv3d(h_conv3a, W_conv3b) + b_conv3b)
            h_pool3 = max_pool3d_2x2x2(h_conv3b)    
    
        # define the 4rd convlutional layer
        with tf.variable_scope('conv4a'):
            W_conv4a = weight_variable([3,3,3,256,512])
            b_conv4a = bias_variable([512])
            h_conv4a = tf.nn.relu(conv3d(h_pool3, W_conv4a) + b_conv4a)
        with tf.variable_scope('conv4b'):
            W_conv4b = weight_variable([3,3,3,512,512])
            b_conv4b = bias_variable([512])
            h_conv4b = tf.nn.relu(conv3d(h_conv4a, W_conv4b) + b_conv4b)
            h_pool4 = max_pool3d_2x2x2(h_conv4b)    
    
        # define the 5rd convlutional layer
        with tf.variable_scope('conv5a'):
            W_conv5a = weight_variable([3,3,3,512,512])
            b_conv5a = bias_variable([512])
            h_conv5a = tf.nn.relu(conv3d(h_pool4, W_conv5a) + b_conv5a)
        with tf.variable_scope('conv5b'):
            W_conv5b = weight_variable([3,3,3,512,512])
            b_conv5b = bias_variable([512])
            h_conv5b = tf.nn.relu(conv3d(h_conv5a, W_conv5b) + b_conv5b)
            h_pool5 = max_pool3d_2x1x1(h_conv5b)    
    
        # define the full connected layer
        with tf.variable_scope('fc6'):
            W_fc6 = weight_variable([int(frmSize[0]/16 * frmSize[1]/16) * 512, 4096])
            b_fc6 = bias_variable([4096])
            h_pool5_flat = tf.reshape(h_pool5, [-1, int(frmSize[0]/16 * frmSize[1]/16) * 512])
            h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc6) + b_fc6)  
            h_fc6_drop = tf.nn.dropout(h_fc6, drop_var) 
            h_fc6_l2norm = tf.nn.l2_normalize(h_fc6_drop,dim=1)
    
        # define the full connected layer fc7
        with tf.variable_scope('fc7'):
            W_fc7 = weight_variable([4096, 4096])
            b_fc7 = bias_variable([4096])
            h_fc7 = tf.nn.relu(tf.matmul(h_fc6_l2norm, W_fc7) + b_fc7)
            h_fc7_drop = tf.nn.dropout(h_fc7, drop_var)
        
        return h_fc7_drop

class Classifier:
    @staticmethod
    def softmax(features,numOfClasses):
        # softmax
        features_l2norm = tf.nn.l2_normalize(features,dim=1)
        with tf.variable_scope('fc7'):
            W_sm = weight_variable([4096, numOfClasses])
            b_sm = bias_variable([4096])
            y_conv = tf.matmul(features_l2norm, W_sm) + b_sm 
        
        return y_conv
        
