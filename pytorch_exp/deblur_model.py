import torch 
import torchvision
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from scipy import io
from random import randint
import time
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image

# https://github.com/milesial/Pytorch-UNet 

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
        return min(len(self.addrs_s), len(self.addrs_b))

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),  
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),  
            nn.ReLU(True),
            nn.Conv2d(64, 32, 5, stride=1, padding=2),  
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(8, 1, 3, stride=1, padding=1)  
        )
        

    def forward(self, x):
        x = self.cnn(x)
        return x


class UNET_Model(nn.Module):
    def __init__(self):
        super(UNET_Model, self).__init__()

    def forward(self, x):
        return x

