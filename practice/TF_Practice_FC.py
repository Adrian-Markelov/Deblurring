import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# Getting the data
from tensorflow.examples.tutorials.mnist import input_data





data = input_data.read_data_sets('data/MNIST', one_hot=True)

img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

training_size = len(data.train.labels)
training_batch_size = 64


# Graph Helper Functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape))

def new_biases(length):
    return tf.Variable(tf.truncated_normal([length]))


def new_conv_layer(input,num_input_channels,filter_size,num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,filter=weights, strides=[1,1,1,1], padding='SAME')
    layer += biases
    if(use_pooling):
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides= [1,2,2,1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights
    

def new_fc_layer(layer_prev, num_inputs, num_outputs, use_relu=True):
    
    weights = new_weights(shape=[num_inputs, num_outputs])
    bias = new_bias(length=num_outputs)
    
    layer_out = tf.add(tf.matmult(layer_prev, weights), bias)
    if(use_relu):
        layer_out = tf.nn.relu(layer_out)
    return layer_out


# network topology
layer_1_size = 1500
layer_2_size = 1500
layer_3_size = 1500
layer_output_size = num_classes

# Build the TF Graph
x_flat = tf.placeholder(tf.float32, shape=[None, img_size_flat], name= 'x_flat')

x = tf.reshape(x_flat, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true)

layer_1 = new_fc_layer(x_flat, img_size_flat, layer_1_size)
layer_2 = new_fc_layer(layer_1, layer_1_size, layer_2_size)
layer_3 = new_fc_layer(layer_2, layer_2_size, layer_3_size)

layer_output = new_fc_layer(layer_3, layer_3_size, layer_output_size)

y_pred = tf.nn.softmax(layer_output)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_output, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_pred_vec = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_pred_vec, tf.float32))










session = tf.Session()

session.run(tf.global_variable_initializer())

saver = tf.train.saver()


# Training Routine (optimize)

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    num_batches = training_size/training_batch_size
    for batch in range(num_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
        epoch_loss += c
    print('Epoch',epoch, 'completed out of ', epochs, 'loss: ', epoch_loss)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,'float'))

print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


saver.save(session, './')


