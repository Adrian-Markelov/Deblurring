
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
plt.ioff()





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
        plt.imshow(imgs_recon[i-2,:,:,0], cmap='gray')

plt.savefig('sample_encodings.png')
plt.close(fig)







