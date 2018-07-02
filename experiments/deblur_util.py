import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import io
from random import randint
from PIL import Image


def get_next_batch(patch_idx, batch_size, num_imgs, num_patches_per_img, data_dir, kernels, load_mem=False, data=None):
    y = None # y is the set of sharp patches
    if(load_mem):
        y = data['y']
     
    num_patches = num_imgs*num_patches_per_img
    num_kernels = kernels.shape[2] # kernels = [41,41, num_kernels]
    
    x_batch = []
    y_flat_true_batch = []
    k_batch = []
    
    patch_idx = patch_idx
    img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
    p_idx = np.mod(patch_idx, num_patches_per_img)
    batch_size = min(batch_size, num_patches-patch_idx)
    for i in range(batch_size):
        # Select a kernel
        k_idx = randint(0, num_kernels-1)
        img_s = None
        
        # Load from memory
        # !!!!! MAKE SURE YOU MAKE THESE IMAGES GRAYSCALE !!!!!
        if(load_mem):
            img_s = y[patch_idx]
        # Loading from disk
        else:
            patch_file = data_dir + '/patch_'+str(img_idx)+'_'+str(p_idx)+'.jpg'
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
        
        # go to next image
        patch_idx = patch_idx + 1
        img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
        p_idx = np.mod(patch_idx, num_patches_per_img)

    x_batch = np.array(x_batch).reshape((batch_size, img_s.shape[0], img_s.shape[0], 1))        
   
    return x_batch, np.array(y_flat_true_batch), np.array(k_batch)
