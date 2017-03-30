import multiprocessing
import numpy as np
import tensorflow as tf
import time
import sys
import ucf101
import videoPreProcess as vpp

if __name__ == "__main__":
    arr = np.random.randn(2,5)
    print(arr)
    arrNorm = tf.nn.l2_normalize(arr,dim=1)
    mean_arr = tf.reduce_mean(arr)
    print(np.mean(arr))
    sess = tf.InteractiveSession()
    with sess.as_default():
        nr = sess.run(mean_arr)
    print(arrNorm)
    print(nr)
    
    