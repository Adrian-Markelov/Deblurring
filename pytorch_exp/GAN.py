
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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






def train(epochs):
    
    d_learning_rate = 2e-4  # 2e-4
    g_learning_rate = 2e-4
    optim_betas = (0.9, 0.999)
    
    d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
    g_steps = 1
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get The Data
    train_loader, img_b_val, img_s_val = setup_data(device)

    
    G_criterion = nn.MSELoss()
    D_criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    
    G = Generator()
    D = Discriminator()
    
    
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
    
    train_loss_log = []
    for epoch in range(epochs):
        i = 0
        for data in train_loader:
            if(i%50 == 0): 
                print('Batch: %d'%i)

            img_b, img_s = data
            img_b = Variable(img_b).to(device)
            img_s = Variable(img_s).to(device)
                        
            for d_index in range(d_steps):
                # 1. Train D on real+fake
                D.zero_grad()

                #  1A: Train D on real
                d_real_decision = D(img_s)
                d_real_error = D_criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params

                #  1B: Train D on fake
                img_s_pred = G(img_b).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(img_s_pred)
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
                d_fake_error.backward()
                d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            for g_index in range(g_steps):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                G.zero_grad()

                img_s_pred = G(img_b) 
                dg_fake_decision = D(img_s_pred)
                g_error = D_criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine
                g_error += G_criterion(img_s, img_s_pred)
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters
        i +=1
    
    G_model_file = 'models/simple_GEN_E'+str(epochs)+'.pth'
    D_model_file = 'models/simple_DES_E'+str(epochs)+'.pth'
    torch.save(G.state_dict(), G_model_file)
    torch.save(D.state_dict(), D_model_file)


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    
    train(epochs)

