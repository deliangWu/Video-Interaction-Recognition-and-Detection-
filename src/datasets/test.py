import multiprocessing
import numpy as np
import tensorflow as tf
import time
import sys
import ucf101
import videoPreProcess as vpp

if __name__ == "__main__":
    ph_input = tf.placeholder(dtype=tf.int32, shape=[None, 1])
    print(ph_input)
    h = tf.zeros([1, 2], dtype=tf.int32)  # ...or some other tensor of shape [1, 2]
    print(h)
    
    # Get the number of rows in the fed value at run-time.
    ph_num_rows = tf.shape(ph_input)[0]
    print(ph_num_rows)
    
    # Makes a `ph_num_rows x 2` matrix, by tiling `h` along the row dimension.
    h_tiled = tf.tile(h, tf.stack([ph_num_rows, 1]))
    print(h_tiled)
    
    result = tf.concat(1,[ph_input, h_tiled,ph_input,ph_input])        
    print(result)