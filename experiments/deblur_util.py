import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import io
from random import randint
from PIL import Image
import pickle

def build_new_super_batch(super_batch, patch_idx, batch_size, num_imgs, num_patches_per_img, data_dir, kernels):

    super_batch_size = x_super_batch.shape[0] 
    sb_idx = int(patch_idx/super_batch_size)
    patch_idx = sb_idx*super_batch_size
 
    num_kernels = kernels.shape[2] # kernels = [41,41, num_kernels]
    img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
    p_idx = np.mod(patch_idx, num_patches_per_img)

    x_super_batch = super_batch[0]
    y_super_batch = super_batch[1]
    k_super_batch = super_batch[2]
 
    for i in range(super_batch_size):
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
        x_super_batch[i,:,:,0] = np.copy(img_s)
        y_super_batch[i,:] = np.copy(img_b)
        k_super_batch[i,:,:] = np.copy(k)
        
        # go to next image
        patch_idx = patch_idx + 1
        img_idx = min(int(np.floor(patch_idx/num_patches_per_img)), num_imgs-1)
        p_idx = patch_idx%num_patches_per_img

    return super_batch, patch_idx


def get_next_super_batch(super_batch, patch_idx, batch_size):
    # Super Batching
    x_super_batch = super_batch[0]
    y_super_batch = super_batch[1]
    k_super_batch = super_batch[2]
    super_batch_size = x_super_batch.shape[0]
    patch_sb_idx = patch_idx%super_batch_size
    x_batch = x_super_batch[patch_idx:patch_sb_idx+batch_size,:,:,:]
    y_flat_true_batch = y_super_batch[patch_sb_idx:patch_idx+batch_size,:]
    k_batch = k_super_batch[patch_idx:patch_sb_idx+batch_size,:,:]
    patch_idx += batch_size
    return super_batch, patch_idx, x_batch, y_flat_true_batch, k_batch

def get_next_batch(super_batch, patch_idx, batch_size, num_imgs, num_patches_per_img, data_dir, kernels, load_mem=False, data=None):
    if(super_batch == None):
        with open('../../data/training_super_batch_'+ str(0) +'.pickle', 'wb') as handle:
            sb_o = pickle.load(handle)
            super_batch = sb_o['super_batch']
    num_patches = num_imgs*num_patches_per_img
    batch_size = min(batch_size, num_patches-patch_idx)
    # if current super batch can fit another batch take it
    if(patch_idx%super_batch_size+batch_size < super_batch_size):
        return get_next_super_batch(super_batch, patch_idx, batch_size)
    
    super_batch, patch_idx = build_new_super_batch(super_batch, patch_idx, batch_size, num_imgs, num_patches_per_img, data_dir, kernels)
    return get_next_super_batch(super_batch, patch_idx, batch_size)
# REMEMBER TO RESET THE patch_idx on edge case!!!



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


super_batch_size = 100000
num_imgs = 12500
num_patches_per_img = 40
training_patches_dir = '../../data/VOC2012_patches/training'
kernels_file = '../../data/kernels/train_kernels.mat'
o = io.loadmat(kernels_file)
kernels = o['kernels']

save_all_super_batches(super_batch_size, num_imgs, num_patches_per_img, training_patches_dir, kernels)







