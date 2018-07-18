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

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 2, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 2, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0),  # b, 8, 15, 15
            nn.ReLU(True)
        )
        
        #print(self.encoder.shape)
        #print(self.encoder.shape)

    def forward(self, x):
        #print('forward')
        #print(x.shape)
        x = self.encoder(x)
        #print('encoder')
        #print(x.shape)
        x = self.decoder(x)
        return x

    
    
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 3
batch_size = 64
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
        
        i +=1

    # ===================log========================
    train_loss_log.append(loss.data[0])
    valid_loss_log.append(loss_val.data[0])
    print('epoch [{}/{}], loss:{:.4f}, loss_valid: {:.4f}'.format(epoch+1, num_epochs, loss.data[0], loss_val.data[0]))

torch.save(model.state_dict(), './deblur_model.pth') 





'''voc_test_dataset = VOC_Dataset(MODE='test')
test_loader = torch.utils.data.DataLoader(dataset=voc_test_dataset,
                                           batch_size=16)
for data in test_loader:
    img_b,_ = data
    img_b = Variable(img_b).to(device)
    output = model(img_b)
    img_b = img_b.cpu().data.numpy()
    img_s_pred = output.cpu().data.numpy()
    
'''





    

