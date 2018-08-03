import torch 
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import time
import numpy as np
import glob
from PIL import Image

import sys
sys.path.insert(0,'./')
from unet_parts import *

# https://github.com/milesial/Pytorch-UNet 

class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, MODE):
        # TODO
        # 1. Initialize file paths or a list of file names
        self.path = '../../data/VOC_patches/'
         
        #self.patch_size_original = 105
        self.patch_size = 128
        
        self.kernels_path = '../../data/kernels/'
        self.kernels = None
        self.addrs_s = None
        self.addrs_b = None
        n_imgs = None
        
        if(MODE=='train'):
            self.addrs_s = glob.glob(self.path+'training/sharp/*.jpg')
            self.addrs_b = glob.glob(self.path+'training/blury/*.jpg')
            n_imgs = len(self.addrs_s)
            print('num_imgs: %d'%n_imgs)
            self.addrs_s = self.addrs_s[0:int(0.8*n_imgs)]
            self.addrs_b = self.addrs_b[0:int(0.8*n_imgs)]
        elif(MODE== 'test'):
            self.addrs_s = glob.glob(self.path+'testing/sharp/*.jpg')
            self.addrs_b = glob.glob(self.path+'testing/blury/*.jpg')
            n_imgs = len(self.addrs_s)
        else:
            self.addrs_s = glob.glob(self.path+'training/sharp/*.jpg')
            self.addrs_b = glob.glob(self.path+'training/blury/*.jpg')
            n_imgs = len(self.addrs_s)
            self.addrs_s = self.addrs_s[int(0.8*n_imgs):n_imgs]
            self.addrs_b = self.addrs_b[int(0.8*n_imgs):n_imgs]
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
        
 
        img_b = img_b.reshape([1, self.patch_size, self.patch_size])
        img_s = img_s.reshape([1, self.patch_size, self.patch_size])
        return img_b, img_s
        
    def __len__(self):
        return min(len(self.addrs_s), len(self.addrs_b))

class RES_CNN_Model(nn.Module):
    def __init__(self):
        super(RES_CNN_Model, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv_4 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv_5 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.conv_6 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(8, 1, 3, stride=1, padding=1)  
        
        

    def forward(self, x):
        l1 = F.relu(self.conv_1(x))
        l2 = F.relu(self.conv_2(l1))
        l3 = F.relu(self.conv_3(l2))
        l4 = F.relu(self.conv_4(l3))
        l5 = F.relu(self.conv_5(l4))
        l6 = F.relu(self.conv_6(l2+l5))
        l7 = F.relu(self.conv_7(l1+l6))
        l8 = F.relu(self.conv_8(l7))
        out = torch.abs(l8-x)
        return out



class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv_4 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv_5 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.conv_6 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(8, 1, 3, stride=1, padding=1)             

    def forward(self, x):
        l1 = F.relu(self.conv_1(x))
        l2 = F.relu(self.conv_2(l1))
        l3 = F.relu(self.conv_3(l2))
        l4 = F.relu(self.conv_4(l3))
        l5 = F.relu(self.conv_5(l4))
        l6 = F.relu(self.conv_6(l5))
        l7 = F.relu(self.conv_7(l6))
        out = F.relu(self.conv_8(l7))
        return out


    
class UNET_Model(nn.Module):
    def __init__(self):
        super(UNET_Model, self).__init__()
        self.inc = inconv(1, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 8)
        self.outc = outconv(8, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv_4 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv_5 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.conv_6 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(8, 1, 3, stride=1, padding=1)  

    def forward(self, x):
        l1 = F.relu(self.conv_1(x))
        l2 = F.relu(self.conv_2(l1))
        l3 = F.relu(self.conv_3(l2))
        l4 = F.relu(self.conv_4(l3))
        l5 = F.relu(self.conv_5(l4))
        l6 = F.relu(self.conv_6(l5))
        l7 = F.relu(self.conv_7(l6))
        l8 = F.relu(self.conv_8(l7))
        out = torch.abs(l8-x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv_4 = nn.Conv2d(32, 16, 5, stride=1, padding=2)
        self.conv_5 = nn.Conv2d(16, 1, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(16384, 5000)
        self.fc2 = nn.Linear(5000, 1)

    def forward(self, x):
        l1 = F.relu(self.conv_1(x))
        l2 = F.relu(self.conv_2(l1))
        l3 = F.relu(self.conv_3(l2))
        l4 = F.relu(self.conv_4(l3))
        l5 = F.relu(self.conv_5(l4))
        l5 = l5.view(-1, 16384)
        l6 = F.relu(self.fc1(l5))
        l7 = F.relu(self.fc2(l6))
        return torch.sigmoid(l7)
