
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1,'../common')
import common


def weight_variable(shape):
    return tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer(stddev=1.1))

def bias_variable(shape):
    return tf.get_variable('bias',shape,initializer=tf.constant_initializer(0.1))

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1,1], padding='SAME')

def conv3d_transpose(x, W,output_shape):
    return tf.nn.conv3d_transpose(x, W, output_shape=output_shape, strides=[1, 1, 1, 1,1], padding='SAME')

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

def unpool3d_2x2x2(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    :param value: A Tensor of shape [b, d, w, h, ch]
    :return: A Tensor of shape [b, 2*d, 2*w, 2*h, ch]
    """
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat([out, out],i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

def unpool3d_1x2x2(value):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    :param value: A Tensor of shape [b, d, w, h, ch]
    :return: A Tensor of shape [b, 1*d, 2*w, 2*h, ch]
    """
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        if i == 1:
            out = out
        else:
            out = tf.concat([out, out],i)
    out_size = [-1,sh[1]] + [s * 2 for s in sh[2:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

def unpool3d_4x2x2(value):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    :param value: A Tensor of shape [b, d, w, h, ch]
    :return: A Tensor of shape [b, 4*d, 2*w, 2*h, ch]
    """
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        if i == 1:
            out = tf.concat(i, [out, out])
            out = tf.concat(i, [out, out])
        else:
            out = tf.concat([out, out],i)
    out_size = [-1,sh[1] * 4] + [s * 2 for s in sh[2:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

def dConv(featureIn,output_shape,out_channels,in_channels,name):
    with tf.variable_scope(name):
        W_conv = weight_variable([3,3,3,out_channels,in_channels])
        b_conv = bias_variable([in_channels])
        print(W_conv.name)
        unPool = unpool3d_1x2x2(featureIn)
        #unBias = tf.nn.relu(unPool - b_conv)
        unBias = unPool
        unConv = conv3d_transpose(unBias, W_conv, output_shape=output_shape)
    return unConv
        
    
    


class FeatureDescriptor:
    @staticmethod
    def c3d(x,frmSize,drop_var, nof_conv1 = 64, nof_conv2 = 128, nof_conv3 = 256, nof_conv4 = 256, noo_fc6 = 4096, noo_fc7 = 4096, rl = 1):
        with tf.device(common.Vars.dev[0]):
            # define the first convlutional layer
            with tf.variable_scope('conv1') as scope:
                numOfFilters_conv1 = nof_conv1 
                W_conv1 = weight_variable([3,3,3,frmSize[2],numOfFilters_conv1])
                b_conv1 = bias_variable([numOfFilters_conv1])
                h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1)
                h_pool1 = max_pool3d_1x2x2(h_conv1)
                print(W_conv1.name)
        
            # define the second convlutional layer
            with tf.variable_scope('conv2'):
                numOfFilters_conv2 = nof_conv2 
                W_conv2 = weight_variable([3,3,3,numOfFilters_conv1,numOfFilters_conv2])
                b_conv2 = bias_variable([numOfFilters_conv2])
                h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = max_pool3d_2x2x2(h_conv2)    
    
            # define the 3rd convlutional layer
            with tf.variable_scope('conv3a'):
                numOfFilters_conv3 = nof_conv3 
                W_conv3a = weight_variable([3,3,3,numOfFilters_conv2,numOfFilters_conv3])
                b_conv3a = bias_variable([numOfFilters_conv3])
                h_conv3a = tf.nn.relu(conv3d(h_pool2, W_conv3a) + b_conv3a)
                h_pool3 = max_pool3d_2x2x2(h_conv3a)    
        
            # define the 4rd convlutional layer
            with tf.variable_scope('conv4a'):
                numOfFilters_conv4 = nof_conv4
                W_conv4a = weight_variable([3,3,3,numOfFilters_conv3,numOfFilters_conv4])
                b_conv4a = bias_variable([numOfFilters_conv4])
                h_conv4a = tf.nn.relu(conv3d(h_pool3, W_conv4a) + b_conv4a)
                h_pool4 = max_pool3d_4x2x2(h_conv4a)    
        
            # define the full connected layer
            with tf.variable_scope('fc6'):
                numOfOutputs_fc6 = noo_fc6
                W_fc6 = weight_variable([int(frmSize[0]/16 * frmSize[1]/16) * numOfFilters_conv4, numOfOutputs_fc6])
                b_fc6 = bias_variable([numOfOutputs_fc6])
                h_pool4_flat = tf.reshape(h_pool4, [-1, int(frmSize[0]/16 * frmSize[1]/16) * numOfFilters_conv4])
                h_fc6 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc6) + b_fc6)  
                h_fc6_drop = tf.nn.dropout(h_fc6, drop_var) 
        
        with tf.device(common.Vars.dev[-1]):
            # define the full connected layer fc7
            with tf.variable_scope('fc7'):
                numOfOutputs_fc7 = noo_fc7 
                W_fc7 = weight_variable([numOfOutputs_fc6, numOfOutputs_fc7])
                b_fc7 = bias_variable([numOfOutputs_fc7])
                h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)
                h_fc7_drop = tf.nn.dropout(h_fc7, drop_var)
                #h_fc7_l2norm = tf.nn.l2_normalize(h_fc7_drop,dim=1)
        if rl == 1:
            return h_pool1
        elif rl == 2:
            return h_pool2
        elif rl == 3:
            return h_pool3
        elif rl == 4:
            return h_pool4
        else: 
            return h_fc7_drop
    
    @staticmethod
    def c3d_v(featureIn,frmSize, nof_conv1 = 64, nof_conv2 = 128, nof_conv3 = 256, nof_conv4 = 256):
        with tf.device(common.Vars.dev[0]):
            # define the first convlutional layer
            featureIn4 = featureIn
            output_shape4 = [1,4,int(frmSize[0]/8),int(frmSize[1]/8),nof_conv3]
            unConv4 = dConv(featureIn4, output_shape4, nof_conv3, nof_conv4, name = 'conv4a')
            
            featureIn3 = unConv4
            output_shape3 = [1,8,int(frmSize[0]/4),int(frmSize[1]/4),nof_conv2]
            unConv3 = dConv(featureIn3, output_shape3, nof_conv2, nof_conv3, name = 'conv3a')
            
            featureIn2 = unConv3
            output_shape2 = [1,16,int(frmSize[0]/2),int(frmSize[1]/2),nof_conv1]
            unConv2 = dConv(featureIn2, output_shape2, nof_conv1, nof_conv2, name = 'conv2')
            
            featureIn1 = unConv2
            output_shape1 = [1,16,frmSize[0],frmSize[1],frmSize[2]]
            unConv1 = dConv(featureIn1, output_shape1, frmSize[2], nof_conv1, name = 'conv1')
        return unConv1
        
