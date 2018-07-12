import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import io
from random import randint
from PIL import Image
import pickle


# Graph Helper Functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape))

def new_biases(length):
    return tf.Variable(tf.truncated_normal(shape=[length]))


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, name=None):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    if use_relu:
        layer = tf.matmul(input, weights) + biases
        layer = tf.nn.relu(layer, name=name)
    else:
        layer = tf.add(tf.matmul(input, weights), biases, name=name)
    return layer


def new_conv_layer(input,num_input_channels,filter_size,num_filters, name, stride=2,use_pooling=True):
    
    layer = tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=filter_size, strides=1, padding='same')
    if(use_pooling):
        layer = tf.nn.max_pool(value=layer, ksize=[1,stride,stride,1], strides= [1,stride,stride,1], padding='SAME')
    layer = tf.nn.relu(layer, name=name)
    return layer


# still might be experimental not sure if backprop works on this
def new_conv_up_layer(input,num_input_channels,filter_size,num_filters, name):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    
    input_shape = input.get_shape().as_list()
    input_img_size = int(input_shape[1])
    output_img_shape = np.array([input_img_size*2, input_img_size*2], np.int32)
    
    upsample_imgs = tf.image.resize_images(images=input, size=output_img_shape,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    layer = tf.layers.conv2d(inputs=upsample_imgs, filters=num_filters, kernel_size=(filter_size,filter_size), padding='same')
    layer = tf.nn.relu(layer, name=name)
    return layer


# This works only if you use the TF layers module for some reason there is likely something different you 
# have to model personalized weights for this
def new_conv_trans_layer(input,num_input_channels,filter_size,num_filters, name, stride=2):
    layer = tf.layers.conv2d_transpose(inputs=input,filters=num_filters, kernel_size=[2,2], strides=2, name=name)
    layer = tf.nn.relu(layer, name=name)
    return layer




#######################################
###### BATCHING HELPER FUNCTIONS ######
#######################################

def get_next_super_batch(super_batch, patch_idx, batch_size, new_load):
    
    if(new_load==False):
        x_super_batch = super_batch[0]
        y_super_batch = super_batch[1]
        k_super_batch = super_batch[2]
        super_batch_size = x_super_batch.shape[0]
        patch_sb_idx = patch_idx%super_batch_size
        x_batch = x_super_batch[patch_idx:patch_sb_idx+batch_size,:,:,:]
        y_flat_true_batch = y_super_batch[patch_sb_idx:patch_idx+batch_size,:]
        k_batch = k_super_batch[patch_idx:patch_sb_idx+batch_size,:,:]
        patch_idx += batch_size
        return super_batch, patch_idx, x_batch, y_true_batch, k_batch
    
    super_batch_size = x_super_batch.shape[0] 
    sb_idx = int(patch_idx/super_batch_size)
    patch_idx = sb_idx*super_batch_size    
    with open('../../data/training_super_batch_'+ str(sb_idx) +'.pickle', 'wb') as handle:
        sb_o = pickle.load(handle)
        super_batch = sb_o['super_batch']
    
    return get_next_super_batch(super_batch, patch_idx, batch_size, new_load=False)

def get_next_batch_sb(super_batch, patch_idx, batch_size, num_imgs, num_patches_per_img, data_dir, kernels, load_mem=False, data=None):
    if(super_batch == None):
        with open('../../data/training_super_batch_'+ str(0) +'.pickle', 'wb') as handle:
            sb_o = pickle.load(handle)
            super_batch = sb_o['super_batch']
    num_patches = num_imgs*num_patches_per_img
    batch_size = min(batch_size, num_patches-patch_idx)
    # if current super batch can fit another batch take it
    if(patch_idx%super_batch_size+batch_size < super_batch_size):
        return get_next_super_batch(super_batch, patch_idx, batch_size, new_load=False)
    return get_next_super_batch(super_batch, patch_idx, batch_size, new_load=True)


def get_next_batch(data, patch_idx, batch_size):
    x = data[0]
    y = data[1]
    k = data[2]
    
    return x[patch_idx:patch_idx+batch_size,:,:,0], y[patch_idx:patch_idx+batch_size,:,:,0], k[patch_idx:patch_idx+batch_size,:,:,0]




##############################
##### ONE TIME ROUTINES ######
##############################

def save_super_batch(super_batch_size, patch_idx, batch_size, num_patches_per_img, data_dir, kernels):
    sb_idx = int(patch_idx/super_batch_size)
    patch_idx = sb_idx*super_batch_size

    img_size = 105
    img_size_flat = img_size*img_size

    kernel_size = 41
    num_kernels = kernels.shape[2] # kernels = [41,41, num_kernels]
    img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
    

    p_idx = np.mod(patch_idx, num_patches_per_img)   
    x_super_batch = np.zeros((super_batch_size, img_size, img_size, 1))
    y_super_batch = np.zeros((super_batch_size, img_size_flat))
    k_super_batch = np.zeros((super_batch_size, kernel_size, kernel_size))
    for i in range(super_batch_size):
        if(i%50==0): print('idx: %d'%i)
        # Select a kernel
        k_idx = randint(0, num_kernels-1)
        
        patch_file = data_dir + '/patch_'+str(img_idx)+'_'+str(p_idx)+'.jpg'
        pil_img_o = Image.open(patch_file).convert('L')
        img_s = np.asarray(pil_img_o)
        
        # Get kernel and generate blury image
        k = kernels[:,:,k_idx] 
        img_b = signal.convolve2d(img_s, k, mode='same')
        
        img_flat_size = img_s.shape[0]**2
        img_s_flat = img_s.reshape((img_flat_size))
        
        # add x,y,k to batch
        x_super_batch[i,:,:,0] = np.copy(img_b)
        y_super_batch[i,:] = np.copy(img_s_flat)
        k_super_batch[i,:,:] = np.copy(k)
        
        # go to next image
        patch_idx = patch_idx + 1
        img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
        p_idx = patch_idx%num_patches_per_img

    super_batch = (x_super_batch, y_super_batch, k_super_batch)
    with open('../../data/training_super_batch_'+ str(sb_idx) +'.pickle', 'wb') as handle:
        pickle.dump(super_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Email rick about batching processes
# 
def save_all_super_batches(super_batch_size, num_imgs, num_patches_per_img, data_dir, kernels):

    num_patches = 500000
    patch_idx = 0
    while(patch_idx < num_patches):
        
        save_super_batch(super_batch_size, patch_idx, num_imgs, num_patches_per_img, data_dir, kernels)
        patch_idx += super_batch_size

    return


def save_all_data(num_patches, num_imgs, num_patches_per_img, data_dir, kernels):
    
    img_size = 105
    kernel_size = 41
    num_kernels = kernels.shape[2] # kernels = [41,41, num_kernels]
    img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
    patch_idx = 0

    p_idx = 0
    x = np.zeros((num_patches, img_size, img_size, 1))
    y = np.zeros((num_patches, img_size, img_size, 1))
    k = np.zeros((num_patches, kernel_size, kernel_size, 1))
    for i in range(num_patches):
        if(i%50==0): print('idx: %d'%i)
        # Select a kernel
        k_idx = randint(0, num_kernels-1)
        
        patch_file = data_dir + '/patch_'+str(img_idx)+'_'+str(p_idx)+'.jpg'
        pil_img_o = Image.open(patch_file).convert('L')
        img_s = np.asarray(pil_img_o)
        
        # Get kernel and generate blury image
        k = kernels[:,:,k_idx] 
        img_b = signal.convolve2d(img_s, k, mode='same')
        
        # add x,y,k to batch
        x[i,:,:,0] = np.copy(img_b)
        y[i,:,:,0] = np.copy(img_s)
        k[i,:,:,0] = np.copy(k)
        
        # go to next image
        patch_idx = patch_idx + 1
        img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
        p_idx = patch_idx%num_patches_per_img

    data = (x, y, k)
    if(training):
        with open('../../data/training_data_small'+'.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('../../data/testing_data_small'+'.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        

num_patches_training = 5#500000
num_imgs_training = 12500
num_patches_per_img_training = 40
training_patches_dir = '../../data/VOC2012_patches/training'

num_patches_testing = 3#3000
num_imgs_testing = 75
num_patches_per_img_testing = 40
testing_patches_dir =  '../../data/VOC2012_patches/testing'


kernels_file = '../../data/kernels/train_kernels.mat'
o = io.loadmat(kernels_file)
kernels = o['kernels']

save_all_data(num_patches_training, num_imgs_training, num_patches_per_img_training, training_patches_dir, kernels)
save_all_data(num_patches_testing,  num_imgs_testing,  num_patches_per_img_testing,  testing_patches_dir,  kernels)





