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

import sys
sys.path.insert(0,'./')
from deblur_model import *



model = CNN_Model()
model.load_state_dict(torch.load('conv_autoencoder.pth'))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
