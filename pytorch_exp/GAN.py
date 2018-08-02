
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 1     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 1    # size of generated output vector
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1



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
        l6 = F.relu(self.conv_6(l2+l5))
        l7 = F.relu(self.conv_7(l1+l6))
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
        self.fc1 = 
        self.fc2 = 

    def forward(self, x):
        l1 = F.relu(self.conv_1(x))
        l2 = F.relu(self.conv_2(l1))
        l3 = F.relu(self.conv_3(l2))
        l4 = F.relu(self.conv_4(l3))
        l5 = F.relu(self.conv_5(l4))
        l5 = l5.view(-1, 16384)
        l6 = F.relu(self.fc1(l5))
        l7 = F.relu(self.fc1(l6))
        return 

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)

d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if epoch % print_interval == 0:
        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                            extract(d_real_error)[0],
                                                            extract(d_fake_error)[0],
                                                            extract(g_error)[0],
                                                            stats(extract(d_real_data)),
                                                            stats(extract(d_fake_data))))


