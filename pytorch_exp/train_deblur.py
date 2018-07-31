import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.ioff()

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import time
import numpy as np
import glob
from PIL import Image

import sys
sys.path.insert(0,'./')
from model_deblur import *



def setup_data(device):
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
    
    return train_loader, img_b_val, img_s_val


def setup_model(model_info)
    learning_rate = 0.001
    
    model = None
    if(model_info == 'CNN'):
        model = CNN_Model().to(device)
    elif(model_info == 'RES_CNN'):
        model = RES_CNN_Model().to(device)
    elif(model_info == 'UNET'):
        model = UNET_Model().to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    return model, optimizer

def spec_loss(loss_model, criterion, output, img_s):
    REGULARIZATION = .0001
    loss = criterion(output, img_s)
    
    if(loss_model == 'TV_REG'):
        reg_loss = REGULARIZATION*(torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) +
                                   torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])))
        loss = loss + reg_loss
    return loss
    
def train(model_info, loss_model, epochs, batch_size):
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Get The Data
    train_loader, img_b_val, img_s_val = setup_data(device)

    # Get The Model
    model, optimizer = setup_model(model_info)
    
    criterion = nn.MSELoss()

    train_loss_log = []
    valid_loss_log = []

    for epoch in range(epochs):
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
            loss = spec_loss(loss_model, criterion, output, img_s)

            # Validation
            output_val = model(img_b_val)
            loss_val = spec_loss(loss_model, criterion, output_val, img_s_val)

            # Update graph
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

        # ===================log========================
        train_loss_log.append(loss.data[0])
        valid_loss_log.append(loss_val.data[0])
        print('epoch [{}/{}], loss:{:.4f}, loss_valid: {:.4f}'.format(epoch+1, num_epochs, loss.data[0], loss_val.data[0]))

    plt.plot(train_loss_log, color='blue', label='training loss')
    plt.plot(valid_loss_log, color='red', label='valid loss')
    plt.legend()

    model_file = 'models/' + model_info + '_'
    loss_file = 'results/loss_' + model_info + '_'
    if(loss_model != None):
        model_file += loss_model
        loss_file += loss_model
    model_file += '_E' + str(epochs) + '.pth'
    loss_file += '_E' + str(epochs) + '.png'
    
    torch.save(model.state_dict(), model_file)
    plt.savefig(loss_file)
    return


if __name__ == "__main__":
    
    mode_info = sys.argv[1]
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    loss_model = None
    if(len(sys.argv) == 5):
        loss_model = sys.argv[4]
    train(model_info, loss_model, epochs, batch_size)








    

