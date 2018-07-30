import glob
from scipy import signal
from scipy import io
from scipy import misc
import numpy as np
from random import randint
from random import shuffle
from PIL import Image
import pickle


voc_dir = '../../../datasets/VOC2012/JPEGImages/'
training_dir = '../../data/VOC_patches/training/'
testing_dir = '../../data/VOC_patches/testing/'

voc_addrs = glob.glob(voc_dir+'*.jpg')
shuffle(voc_addrs)
n_voc = len(voc_addrs)

voc_train_addrs = voc_addrs[0:int(0.8*n_voc)] # validation will be split from the train set
voc_test_addrs = voc_addrs[int(0.8*n_voc):n_voc]

n_train_voc = len(voc_train_addrs)
n_test_voc = len(voc_test_addrs)

#train_kernels_o = io.loadmat('../../data/kernels/train_kernels.mat')
#test_kernels_o = io.loadmat('../../data/kernels/test_kernels.mat')
#train_kernels = train_kernels_o['kernels']
#test_kernels = test_kernels_o['kernels']

train_kernels = None
test_kernels = None

with open('../../data/kernels/train_kernels.pickle', 'rb') as handle:
    train_kernels = pickle.load(handle)

with open('../../data/kernels/test_kernels.pickle', 'rb') as handle:
    test_kernels = pickle.load(handle)




noise_sigma = 2
patch_size_plus = 136
patch_size = 128


n_patches_per_img = 2


def get_patch(img, patch_size):
    img_size_h = img.shape[0]
    img_size_w = img.shape[1]
    h_bound = img_size_h - patch_size
    w_bound = img_size_w - patch_size

    h_corner = randint(0, h_bound-1)
    w_corner = randint(0, w_bound-1)

    patch = img[h_corner:h_corner+patch_size, w_corner:w_corner+patch_size]
    return patch

def make_patches(voc_addrs, kernels, save_path):
    
    n_voc = len(voc_addrs)

    patch_count = 0
    for i in range(n_voc):
        print(patch_count)
        img = Image.open(voc_addrs[i]).convert('L')
        img = np.asarray(img)
        if(img.shape[0] <= patch_size_plus or img.shape[1] <= patch_size_plus):
            continue
        for j in range(n_patches_per_img):
            img_s_plus = get_patch(img, patch_size_plus)
            img_s = img_s_plus[4:132,4:132]
            # Convolve with blur kernel
            n_kernels = kernels.shape[2]
            k_idx = randint(0, n_kernels-1)
            k = kernels[k_idx,:,:]

            img_b = signal.convolve2d(img_s_plus, k, mode='valid')

            # Add white noise
            noise = np.random.randn(patch_size, patch_size) * noise_sigma            
            img_b = np.absolute(img_b + noise)
            # Save Image            
            img_b = img_b.astype(np.uint8)
            img_s = img_s.astype(np.uint8)
            misc.toimage(img_b, cmin=0.0, cmax=...).save(save_path+'blury/patch_{}.jpg'.format(patch_count))
            misc.toimage(img_s, cmin=0.0, cmax=...).save(save_path+'sharp/patch_{}.jpg'.format(patch_count))
            patch_count += 1
    



 # Gen training set
make_patches(voc_train_addrs, train_kernels, training_dir)

# Gen testing set
make_patches(voc_test_addrs, test_kernels, testing_dir)
   



