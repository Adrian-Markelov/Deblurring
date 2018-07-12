import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import io
from random import randint
from PIL import Image
import pickle
import glob


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# CONVERT IMAGES TO 128 *******
def load_image(addr, kernels):
    img_s_o = Image.open(addr).convert('L')
    img_s = np.asarray(img_s_o)
    if img_s is None:
        return None
    img_s_full = np.zeros((128,128))
    img_s_full[11:116, 11:116] = img_s
    img_s = img_s_full
    num_kernels = kernels.shape[2] # kernels = [41,41, num_kernels]
    k = k_idx = randint(0, num_kernels-1)
    k = kernels[:,:,k_idx] 
    img_b = signal.convolve2d(img_s, k, mode='same')
    return img_s, img_b, k
 
def createDataRecord(out_filename, addrs, kernels_file):
    
    # Get the kernel matrix
    o = io.loadmat(kernels_file)
    kernels = o['kernels']
    
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img_s, img_b, k = load_image(addrs[i], kernels)

        if img is None:
            continue

        # Create a feature
        feature = {
            'img_s_raw': _bytes_feature(img_s.tostring()),
            'img_b_raw': _bytes_feature(img_b.tostring()),
            'k_raw': _bytes_feature(k.tostring())
            
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

train_path = '../../data/VOC2012_patches/training/*.jpg'
test_path = '../../data/VOC2012_patches/testing/*.jpg'
train_kernels_file = '../../kernels/train_kernels.mat'
test_kernels_file = '../../kernels/test_kernels.mat'
train_addrs = glob.glob(train_path)
test_addrs = glob.glob(test_path)

    
# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = train_addrs
test_addrs = test_addrs[0:int(0.8*len(test_addrs))]
val_addrs = test_addrs[int(0.8*len(test_addrs)):]


createDataRecord('../../data/TF_data/train.tfrecords', train_addrs, train_kernels_file)
createDataRecord('test.tfrecords',  test_addrs, test_kernels_file)
createDataRecord('val.tfrecords',   val_addrs, test_kernels_file)



