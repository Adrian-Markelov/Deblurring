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



    
    
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
batch_size = 128
learning_rate = 0.001


# You can then use the prebuilt data loader. 
voc_dataset = VOC_Dataset(MODE='train')
train_loader = torch.utils.data.DataLoader(dataset=voc_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
voc_valid_dataset = VOC_Dataset(MODE='valid')
valid_loader = torch.utils.data.DataLoader(dataset=voc_valid_dataset,
                                           batch_size=batch_size)
for data in valid_loader:
    img_b_val, img_s_val = data
    break

img_b_val = Variable(img_b_val).to(device)
img_s_val = Variable(img_s_val).to(device)

model = CNN_Model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
weight_decay=1e-5)

train_loss_log = []
valid_loss_log = []

for epoch in range(num_epochs):
    start = time.time()
    i = 0
    for data in train_loader: 
        if(i%100 == 0): 
            print('batch: %d'%i)
        start_in = time.time()
        
        # Training 
        img_b, img_s = data
        img_b = Variable(img_b).to(device)
        img_s = Variable(img_s).to(device)
        output = model(img_b)
        loss = criterion(output, img_s)
        
        # Validation
        output_val = model(img_b_val)
        loss_val = criterion(output_val, img_s_val)

        # Update graph
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        i += 1

    # ===================log========================
    train_loss_log.append(loss.data[0])
    valid_loss_log.append(loss_val.data[0])
    print('epoch [{}/{}], loss:{:.4f}, loss_valid: {:.4f}'.format(epoch+1, num_epochs, loss.data[0], loss_val.data[0]))

torch.save(model.state_dict(), 'cnn_%d.pth'%num_epochs) 










    

