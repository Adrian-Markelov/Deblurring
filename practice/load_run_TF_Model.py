import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'../../Deblurring/experiments')
import deblur_util
from scipy import io
from PIL import Image
import pickle

session = tf.Session()

saver = tf.train.import_meta_graph('model_2.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))


graph = tf.get_default_graph()
x = graph.get_tensor_by_name('x:0')

output_layer = graph.get_tensor_by_name('output_layer:0')




o = io.loadmat('../../data/kernels/train_kernels.mat')
kernels = o['kernels']
testing_patches_dir = '../../data/VOC2012_patches/testing'

testing_batch_size = 64
num_test_patches = 64
num_testing_imgs = 75
num_patches_per_img = 40

patch_idx = 0


x_batch, y_flat_true_batch, k_batch = deblur_util.get_next_batch(patch_idx, testing_batch_size, num_testing_imgs, num_patches_per_img, testing_patches_dir, kernels)

feed_dict = {x: x_batch}

pred_flat_imgs = session.run(output_layer, feed_dict)

print(pred_flat_imgs.shape)

pred_imgs = []

for i in range(num_test_patches):
    img = pred_flat_imgs[i].reshape((105,105))
    pred_imgs.append(img)

with open('pred_imgs.pickle', 'wb') as handle:
    pickle.dump(pred_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(pred_imgs[0].shape)
img_o = Image.fromarray(pred_imgs[0], mode='L')
img_o.save('pred_img_0.png')




