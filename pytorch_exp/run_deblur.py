
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
from model_deblur import *

import matplotlib
#matplotlib.use('agg')
plt.switch_backend('agg')
plt.ioff()


if __name__ == "__main__":
    
    model_file = sys.argv[1]
    model_info = sys.argv[2]
    
    model = None
    if(model_info == 'CNN'):
        model = CNN_Model()
    elif(model_info == 'RES_CNN'):
        model = RES_CNN_Model()
    elif(model_info == 'UNET'):
        model = UNET_Model()
    elif(model_info == 'GAN'):
        model = Generator()
    
    model.load_state_dict(torch.load(model_file))


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    voc_test_dataset = VOC_Dataset(MODE='test')
    test_loader = torch.utils.data.DataLoader(dataset=voc_test_dataset,
                                               batch_size=64)
    all_inputs = []
    all_outputs = []
    all_true = []
    for data in test_loader:
        img_b,img_s = data
        img_b = Variable(img_b)
        img_s = Variable(img_s)
        output = model(img_b)
        all_inputs.append(img_b.cpu().data.numpy())
        all_outputs.append(output.cpu().data.numpy())
        all_true.append(img_s.cpu().data.numpy())
        break

    img_s_batch = all_true[0]
    img_b_batch = all_inputs[0]
    output_batch = all_outputs[0]


    '''fig = plt.figure(figsize=(20,30))
    fig.add_subplot(1,3,1)
    plt.imshow(img_s_batch[1,0,:,:],cmap='gray')
    fig.add_subplot(1,3,2)
    plt.imshow(img_b_batch[1,0,:,:],cmap='gray')
    fig.add_subplot(1,3,3)
    plt.imshow(output_batch[1,0,:,:],cmap='gray')
    plt.savefig('results/single_deblurs.png')
    plt.close(fig)
    '''

    plot_rows = 6
    plot_cols = 9
    fig = plt.figure(figsize=(20,20))
    for i in range(1,plot_rows*plot_cols+1):
        if(i%3==1):
            img_s = img_s_batch[i,0,:,:]
            fig.add_subplot(plot_rows,plot_cols,i)
            plt.imshow(img_s,cmap='gray')
        elif(i%3==2):
            img_b = img_b_batch[i-1,0,:,:]
            fig.add_subplot(plot_rows,plot_cols,i)
            plt.imshow(img_b,cmap='gray')

        else:
            output = output_batch[i-2,0,:,:]
            fig.add_subplot(plot_rows,plot_cols,i)
            plt.imshow(output, cmap='gray')

    model_info = model_file[7:len(model_file)-4]
    img_file = 'results/deblur_imgs_' + model_info+'.png'
    plt.savefig(img_file)
    plt.close(fig)
