#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from scipy import signal
from scipy import io
import matplotlib.image as mpimg
from random import randint
from PIL import Image



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
                   use_pooling=True):  # Use 2x2 max-pooling.

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

    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def get_batch(batch_size, kernels, load_mem, data=None):
    y = None # y is the set of sharp patches
    if(load_mem):
        y = data['y']
    
    training_patches_dir = '../../../datasets/VOC2012_patches/training'    
    
    num_img = 12500
    num_patches_per_img = 40
    num_patches = num_img*num_patches_per_img
    num_kernels = kernels.shape[2] # kernels = [41,41, num_kernels]
    
    x_batch = []
    y_flat_true_batch = []
    k_batch = []
    
    
    for i in range(batch_size):
        # Select a kernel
        k_idx = randint(0, num_kernels-1)
        img_s = None
        
        # Load from memory
        # !!!!! MAKE SURE YOU MAKE THESE IMAGES GRAYSCALE !!!!!
        if(load_mem):
            patch_idx = randint(0, num_patches-1)
            img_s = y[patch_idx]
        # Loading from disk
        else:
            img_idx = randint(0, num_img-1)
            p_idx = randint(0, num_patches_per_img-1)
            patch_file = training_patches_dir + '/patch_'+str(img_idx)+'_'+str(p_idx)+'.jpg'
            pil_img_o = Image.open(patch_file).convert('L')
            img_s = np.asarray(pil_img_o)
        
        # Get kernel and generate blury image
        k = kernels[:,:,k_idx] 
        img_b = signal.convolve2d(img_s, k, mode='same')
        
        img_flat_size = img_s.shape[0]**2
        img_s_flat = img_s.reshape((img_flat_size))
        
        # add x,y,k to batch
        x_batch.append(img_b)
        y_flat_true_batch.append(img_s_flat)
        k_batch.append(k)

    x_batch = np.array(x_batch).reshape((batch_size, img_s.shape[0], img_s.shape[0], 1))        
   
    return x_batch, np.array(y_flat_true_batch), np.array(k_batch)





    # Data will come in batches 
#data = input_data.read_data_sets('data/MNIST/', one_hot=True)
o = io.loadmat('../../kernels/train_kernels.mat')
kernels = o['kernels']
training_patches_dir = '../../../datasets/VOC2012_patches/training'

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


train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

# Split the test-set into smaller batches of this size.
test_batch_size = 8



# FIX NETWORK OUTPUT TOPOLOGY

## Building the graph for the Neural network with placeholders ONLY

x = tf.placeholder(tf.float32, shape=[train_batch_size, img_size, img_size, num_channels], name='x')

y_flat_true = tf.placeholder(tf.float32, shape=[None, output_size], name='y_true')


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
                         use_relu=False)
y_flat_pred = output

cost = tf.reduce_mean(tf.square(y_flat_pred-y_flat_true))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)




def optimize(optimizer, cost, num_iterations, kernels, data=None, load_mem=False):
    # num_iterations is the number of iters to add onto how many have already been done
    
    saver = tf.train.Saver()

    ## Initializing a session for the neural network
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        
        x_batch, y_flat_true_batch, k_batch = get_batch(train_batch_size, kernels, load_mem, data)
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

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    save_path = saver.save(session, "../../TF_models/model_1.ckpt")


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))



    #print_test_accuracy()
#optimize(optimizer, cost, num_iterations=1, kernels=kernels)
optimize(optimizer, cost, num_iterations=10000, kernels=kernels) # We already performed 1 iteration above.
#print_test_accuracy()
