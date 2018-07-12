import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import numpy as np

from scipy import signal
from scipy import io
from random import randint
from PIL import Image
import pickle

sys.path.insert(0,'./')
from deblur_util import *

kernel_size = 41
img_size = 128
img_size_flat = img_size * img_size

'''training_batch_size = 64
test_batch_size = 64
val_batch_size = 64


# Data info
n_training_patches = 500000   
n_test_patches = 2400
n_valid_patches = 600'''


def parser(record):
    keys_to_features = {
        "img_b_raw": tf.FixedLenFeature([], tf.string),
        "img_s_raw": tf.FixedLenFeature([], tf.string),
        "k_raw": tf.FixedLenFeature([], tf.string)
         }
    parsed = tf.parse_single_example(record, keys_to_features)
    img_b = tf.decode_raw(parsed["img_b_raw"], tf.uint8) 
    img_s = tf.decode_raw(parsed["img_s_raw"], tf.uint8) 
    k = tf.decode_raw(parsed["k_raw"], tf.uint8) 
    img_b = tf.cast(img_b, tf.float32)
    img_s = tf.cast(img_s, tf.float32)
    k = tf.cast(k, tf.float32)
    return img_b, img_s


def input_fn(filenames):
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  '''dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(1024, 1)
  )'''
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, batch_size=64)
  )
  #dataset = dataset.map(parser, num_parallel_calls=12)
  #dataset = dataset.batch(batch_size=4)
  #dataset = dataset.prefetch(buffer_size=2)
  return dataset


def train_input_fn():
    return input_fn(filenames=["../../data/TF_data_tests/train.tfrecords", "../../data/TF_data/test.tfrecords"])

def val_input_fn():
    return input_fn(filenames=["../../data/TF_data_tests/val.tfrecords"])



def model_fn(features, labels, mode, params):

    img_b = features
    img_s = labels
    
    filter_size = 3

    n_channels_x = 1 # size = 128
    # Down-sample
    n_filters_l1 = 32  # size = 64      
    n_filters_l2 = 64  # size = 32       
    n_filters_l3 = 128 # size = 16
    # Up-sample
    n_filters_l4 = 64 # size = 32        
    n_filters_l5 = 32  # size = 64       
    n_output_channels = 1 # size = 128

    img_s = tf.reshape(img_s, [-1,128,128,1])

    img_b = tf.identity(img_b, name="input_tensor")
    img_b = tf.reshape(img_b, [-1, 128, 128, 1])    
    img_b = tf.identity(img_b, name="input_tensor_after")

    # Down-sample: 128,64,32,16 <-> 32,64,128
    conv_layer_1 = new_conv_layer(img_b,          n_channels_x, filter_size, n_filters_l1, name='conv_layer_1')
    conv_layer_2 = new_conv_layer(conv_layer_1, n_filters_l1, filter_size, n_filters_l2, name='conv_layer_2')
    conv_layer_3 = new_conv_layer(conv_layer_2, n_filters_l2, filter_size, n_filters_l3, name='conv_layer_3')
    
    # Up-sample
    conv_layer_4 = new_conv_trans_layer(conv_layer_3, n_filters_l3, filter_size, n_filters_l4, name='conv_layer_4')
    conv_layer_2_4 = tf.concat([conv_layer_2, conv_layer_4], 3)
    conv_layer_5 = new_conv_trans_layer(conv_layer_2_4, n_filters_l2+n_filters_l4, filter_size, n_filters_l5, name='conv_layer_5')
    conv_layer_1_5 = tf.concat([conv_layer_1, conv_layer_5], 3)
    output_layer = new_conv_trans_layer(conv_layer_1_5, n_filters_l1+n_filters_l5, filter_size, n_output_channels, name='output_layer')

    img_s_pred = output_layer


    #print('img_s_pred: %s'%img_s_pred.get_shape())
    #print('img_s: %s'%img_s.get_shape())


    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=img_s_pred)
    else:
        loss = tf.reduce_mean(tf.square(img_s_pred-img_s))

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)
        
    return spec

model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 1e-4},
                               model_dir="./deblur_model/")

count = 0
while (count < 10):
    model.train(input_fn=train_input_fn)#, steps=1000)# steps = num_batches
    result = model.evaluate(input_fn=val_input_fn)
    print('loss: %d'%result['loss'])
    sys.stdout.flush()
    count = count + 1



pred_results = model.predict(input_fn=val_input_fn)

print(pred_results)








