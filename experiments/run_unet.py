

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
plt.ioff()




def test(session, kernels):

    # The starting index for the next batch is denoted i.
    patch_idx = 0
    
    patch_sharp_pred = np.zeros((num_test_patches, img_size_flat))
    
    while patch_idx < num_test_patches:

        # Get the images from the test-set between index i and j.
        patches = deblur_util.get_next_batch(patch_idx, testing_batch_size, num_testing_imgs, num_patches_per_img, tresting_patches_dir, kernels) #data.test.images[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: patches}

        patch_idx_end = min(patch_idx+testing_batch_size, num_test_patches)
        # Calculate the predicted class using TensorFlow.
        patch_sharp_pred[patch_idx:patch_idx_end, :] = session.run(y_flat_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        patch_idx = patch_idx_end







# Parameters for user to set
testing_batch_size = 64
epochs = 10
plot_rows = 4 # num of test image rows to run and plot
plot_cols = 6 # ditto (cols must be even)
add_skip_connect = False # if true will learn ID function in like 2 epochs




# Getting the MNIST data set and logging its parameters
data = input_data.read_data_sets('data/MNIST', one_hot=True)
img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels_x = 1
training_size = len(data.train.labels)



session = tf.Session()

saver = tf.train.import_meta_graph('AE_model/AE_model.meta')
saver.restore(session, tf.train.latest_checkpoint('AE_model'))


graph = tf.get_default_graph()
x = graph.get_tensor_by_name('x:0')

output_layer = graph.get_tensor_by_name('output_layer:0')
conv_layer_1 = graph.get_tensor_by_name('conv_layer_1:0')
conv_layer_2 = graph.get_tensor_by_name('conv_layer_2:0')
conv_layer_3 = graph.get_tensor_by_name('conv_layer_3:0')


# Run the learned network on testing data
flat_imgs,_ = data.test.next_batch(testing_batch_size)
imgs = flat_imgs.reshape((-1,img_size,img_size,1))
feed_dict = {x: imgs}
[imgs_recon, layer_3_imgs] = session.run([output_layer, conv_layer_3], feed_dict)




# Show original images and reconstructed images
fig = plt.figure(figsize=(10,10))
for i in range(1,plot_rows*plot_cols+1):
    if(i%2==1):
        fig.add_subplot(plot_rows,plot_cols,i)
        plt.imshow(imgs[i-1,:,:,0],cmap='gray')
    else:
        fig.add_subplot(plot_rows,plot_cols,i)
        plt.imshow(layer_3_imgs[i-2,:,:,1], cmap='gray')

plt.savefig('sample_encodings.png')
plt.close(fig)