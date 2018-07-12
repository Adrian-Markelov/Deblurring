import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
from scipy import signal
from scipy import io
import matplotlib.image as mpimg
from random import randint
from PIL import Image
import sys
sys.path.insert(0,'./')
from deblur_util *


kernel_size = 41
img_size = 128
img_size_flat = img_size * img_size


filter_size = 3



n_channels_x = 1
# Down-sample
n_filters_l1 = 32         
n_filters_l2 = 64         
n_filters_l3 = 128
# Up-sample
n_filters_l4 = 64         
n_filters_l5 = 32         
n_output_channels = 1


training_batch_size = 64
test_batch_size = 64
val_batch_size = 64


# Data info
n_training_patches = 500000   
n_test_patches = 2400
n_valid_patches = 600


## Building the graph for the Neural network with placeholders ONLY
def build_unet(x, y_true):
    # Down Sampling
    conv_layer_1 = new_conv_layer(x ,num_channels_x,filter_size,num_filters_l1, name='conv_layer_1')
    conv_layer_2 = new_conv_layer(conv_layer_1, num_filters_l1, filter_size, num_filters_l2, name='conv_layer_2')    
    
    # Up Sampling
    conv_layer_3 = new_conv_up_layer(conv_layer_2, num_filters_l2, filter_size, num_filters_l3, name='conv_layer_3')
    # Add conv_layer_1 and conv_layer_3 together (both are (14,14) images)
    
    if(add_skip_connect):
        print('conv_layer_3 in skip connected. Shape should be: [%s + %s]'%(conv_layer_3.get_shape(), conv_layer_1.get_shape()))
        conv_layer_3 = tf.concat([conv_layer_3,conv_layer_1],3) # index 3 is the channel index
        conv_layer_3_shape = conv_layer_3.get_shape().as_list()
        num_filters_l3 = tf.cast(conv_layer_3_shape[3], tf.int32)
    
    output = new_conv_up_layer(conv_layer_3, num_filters_l3, filter_size, num_filters_output, name='output_layer')
    
    # save an optimizer for the given graph above
    cost = tf.reduce_mean(tf.square(output-x))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    cost = tf.reduce_mean(tf.square(y_pred-y_true))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    return cost,optimizer, output




def train(session, optimizer, cost, num_iterations, data):
    start_time = time.time()
    
    for i in range(num_iterations):
        print('iter: %d'%i)
        super_batch = None 
        patch_idx = 0
        for batch_idx in range(int(num_training_patches/training_batch_size)):
            if(i%5==0):
                print('batch idx: %d'%batch_idx)
            
            x_batch, y_true_batch, k_batch = get_next_batch(patch_data, patch_idx, batch_size)
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            
            session.run(optimizer, feed_dict=feed_dict_train)

        if i % 10 == 0:
            cost_val = session.run(cost, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Cost: {1:>6.1%}"
            print(msg.format(i, cost_val))

    end_time = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end_time-start_time)))))






## Initializing a session for the neural network
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()



x = tf.placeholder(tf.float32, shape=[training_batch_size, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, output_size], name='y_flat_true')

cost, optimizer, output = build_unet(x, y_true)

train(session, optimizer, cost, num_iterations=50, kernels=kernels)

save_path = saver.save(session, "../../TF_models/cnn_model_4/model_4")




