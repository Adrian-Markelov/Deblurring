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
import util.py


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

    
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True, 
                   name=None):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases
    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer, name=name)
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True, # Use Rectified Linear Unit (ReLU)?
                 name=None):

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    if use_relu:
        layer = tf.matmul(input, weights) + biases
        layer = tf.nn.relu(layer, name=name)
    else:
        layer = tf.add(tf.matmul(input, weights), biases, name=name)
    return layer



    # Data will come in batches 
#data = input_data.read_data_sets('data/MNIST/', one_hot=True)
o = io.loadmat('../../data/kernels/train_kernels.mat')
kernels = o['kernels']
training_patches_dir = '../../data/VOC2012_patches/training'
testing_patches_dir = '../../data/VOC2012_patches/testing' 
# We know that MNIST images are 28 pixels in each dimension.
img_size = 105

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 36 of these filters.

# Convolutional Layer 3
filter_size3 = 5
num_filters3 = 128



# Fully-connected layer.
fc_size = img_size_flat   # Number of neurons in fully-connected layer.



# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Size of the output sharp patch
output_size = img_size_flat


training_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

# Split the test-set into smaller batches of this size.
test_batch_size = 8


# Training info
num_training_patches = 5
num_training_imgs = 12500
num_patches_per_img = 40    

# Testing Data
num_test_patches = 3000
num_imgs = 75
num_patches_per_uniq_img = 40


## Building the graph for the Neural network with placeholders ONLY

x = tf.placeholder(tf.float32, shape=[training_batch_size, img_size, img_size, num_channels], name='x')

y_flat_true = tf.placeholder(tf.float32, shape=[None, output_size], name='y_flat_true')


layer_conv1, weights_conv1 = \
    new_conv_layer(input=x,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)


layer_flat, num_features = flatten_layer(layer_conv3)


layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=True)

output = new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size,
                         num_outputs=output_size,
                         use_relu=False,
                         name='output_layer')
y_flat_pred = output

cost = tf.reduce_mean(tf.square(y_flat_pred-y_flat_true))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)




def train(session, optimizer, cost, num_iterations, kernels, data=None, load_mem=False):

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    for i in range(num_iterations):

        patch_idx = 0
        for batch_idx in range(int(num_training_patches/training_batch_size)):
            x_batch, y_flat_true_batch, k_batch = get_batch(patch_idx, training_batch_size, num_training_imgs, num_patches_per_img, training_patches_dir, kernels)
            feed_dict_train = {x: x_batch,
                               y_flat_true: y_flat_true_batch}

            session.run(optimizer, feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if i % 5 == 0:
                # Calculate the accuracy on the training-set.
                cost_val = session.run(cost, feed_dict=feed_dict_train)

                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Training Cost: {1:>6.1%}"

                # Print it.
                print(msg.format(i + 1, cost_val))

            patch_idx = patch_idx + training_batch_size

    # Ending time.
    end_time = time.time()

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(end_time-start_time)))))


def test(session, kernels):

    # The starting index for the next batch is denoted i.
    patch_idx = 0
    
    patch_sharp_pred = np.zeros((num_test_patches, img_size_flat))
    
    while patch_idx < num_test_patches:

        # Get the images from the test-set between index i and j.
        patches = get_batch(patch_idx, testing_batch_size, num_testing_imgs, num_patches_per_img, tresting_patches_dir, kernels) #data.test.images[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: patches}

        patch_idx_end = min(patch_idx+testing_batch_size, num_test_patches)
        # Calculate the predicted class using TensorFlow.
        patch_sharp_pred[patch_idx:patch_idx_end, :] = session.run(y_flat_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        patch_idx = patch_idx_end



## Initializing a session for the neural network
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()


train(session, optimizer, cost, num_iterations=1, kernels=kernels)
#test(session, kernels)


save_path = saver.save(session, "../../TF_models/cnn_model_2/model_2")




