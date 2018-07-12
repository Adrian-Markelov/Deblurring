
import tensorflow as tf
import sys
import numpy as np

from scipy import signal
from scipy import io
from random import randint
from PIL import Image
import pickle


def parser(record):
    keys_to_features = {
        "img_b_raw": tf.FixedLenFeature([], tf.string),
        "img_s_raw": tf.FixedLenFeature([], tf.string),
        "k_raw": tf.FixedLenFeature([], tf.string) 
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    img_b = tf.decode_raw(parsed["img_b_raw"], tf.uint8) #<<<------ THIS IS PROB not uint8
    img_s = tf.decode_raw(parsed["img_s_raw"], tf.uint8) #<<<------ THIS IS PROB not uint8
    k = tf.decode_raw(parsed["k_raw"], tf.uint8) #<<<------ THIS IS PROB not uint8
    img_b = tf.cast(img_b, tf.float32)
    img_s = tf.cast(img_s, tf.float32)
    k = tf.cast(k, tf.float32)
    return img_b, img_s, k


def input_fn(filenames):
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  '''dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(1024, 1)
  )'''
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 32)
  )
  #dataset = dataset.map(parser, num_parallel_calls=12)
  #dataset = dataset.batch(batch_size=1000)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset


def train_input_fn():
    return input_fn(filenames=["../../data/TF_data/train.tfrecords", "../../data/TF_data/test.tfrecords"])

def val_input_fn():
    return input_fn(filenames=["../../data/TF_data/val.tfrecords"])






#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

train_dataset = train_input_fn()
val_dataset = val_input_fn()


print('printing out dataset type .......................')
print(train_dataset)
print(val_dataset)



# create a one-shot iterator
iterator = val_dataset.make_one_shot_iterator()
# extract an element
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        val = sess.run(next_element)
        print(val)






