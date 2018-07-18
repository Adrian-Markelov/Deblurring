import matplotlib.pyplot as plt
import torch 
import torchvision
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from scipy import signal
from scipy import io
from random import randint
import time
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image


class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, MODE):
        # TODO
        # 1. Initialize file paths or a list of file names
        self.path = '../../data/VOC2012_patches/'
         
        self.patch_size_original = 105
        self.patch_size = 128
        
        self.kernels_path = '../../data/kernels/'
        self.kernels = None
        self.addrs_s = None
        self.addrs_b = None
        
        if(MODE=='train'):
            self.addrs_s = glob.glob(self.path+'training_sharp/*.jpg')
            self.addrs_b = glob.glob(self.path+'training_blury/*.jpg')
            kernels_file = self.kernels_path + 'train_kernels.mat'
            o = io.loadmat(kernels_file)
            self.kernels = o['kernels']
            print(len(self.addrs_s))

        elif(MODE== 'test'):
            self.addrs_s = glob.glob(self.path+'testing_sharp/*.jpg')
            self.addrs_b = glob.glob(self.path+'testing_blury/*.jpg')
            kernels_file = self.kernels_path + 'test_kernels.mat'
            o = io.loadmat(kernels_file)
            self.kernels = o['kernels']
        else:
            self.addrs_s = glob.glob(self.path+'valid_sharp/*.jpg')
            self.addrs_b = glob.glob(self.path+'valid_blury/*.jpg')
            kernels_file = self.kernels_path + 'test_kernels.mat'
            o = io.loadmat(kernels_file)
            self.kernels = o['kernels']
            
        self.kernels = self.kernels.astype(dtype=np.float32)        
        self.num_kernels = self.kernels.shape[2] # kernels = [41,41, num_kernels]
        return

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img_s = Image.open(self.addrs_s[index]).convert('L')
        img_b = Image.open(self.addrs_b[index]).convert('L')    
        img_s = np.asarray(img_s, dtype=np.float32)
        img_b = np.asarray(img_b, dtype=np.float32)

        #print('getting image')
        #print(self.addrs_s[index])
        #print(img_s.shape)
        #print(img_b.shape)       
        
        if(img_s.shape[0] == 105):
            img_s_full = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            img_s_full[11:116, 11:116] = img_s
            img_s = img_s_full       
 
        img_b = img_b.reshape([1, self.patch_size, self.patch_size])
        img_s = img_s.reshape([1, self.patch_size, self.patch_size])
        return img_b, img_s
        
    def __len__(self):
        return len(self.addrs_s)


voc_test_dataset = VOC_Dataset(MODE='test')
test_loader = torch.utils.data.DataLoader(dataset=voc_test_dataset,
                                           batch_size=16)
all_inputs = []
all_outputs = []
for data in test_loader:
    img_b,img_s = data
    img_b = Variable(img_b).to(device)
    img_s = Variable(img_b).to(device)
    output = model(img_b)
    all_inputs.append(img_b.cpu().data.numpy())
    all_outputs.append(output.cpu().data.numpy())
    
img_b_batch = all_inputs[0]
output_batch = all_outputs[0]

plot_rows = 4
plot_cols = 6
fig = plt.figure(figsize=(10,10))
for i in range(1,plot_rows*plot_cols+1):
    img_b = img_b_batch[i,0,:,:]
    output = output_batch[i,0,:,:]
    if(i%2==1):
        fig.add_subplot(plot_rows,plot_cols,i)
        plt.imshow(img_b,cmap='gray')
    else:
        fig.add_subplot(plot_rows,plot_cols,i)
        plt.imshow(output, cmap='gray')

plt.savefig('sample_deblurs.png')
plt.close(fig)