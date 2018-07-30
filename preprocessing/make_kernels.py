
import numpy as np
from scipy import io
from PIL import Image
import pickle

k_o = io.loadmat('../../data/kernels/train_kernels.mat')
k = k_o['kernels']


k_new = np.zeros((10000,9,9), dtype=np.float32)
for i in range(10000):
    k_new[i,:,:] = k[15:24,15:24,i]
k = k_new
with open('../../data/kernels/train_kernels.pickle', 'wb') as handle:
    pickle.dump(k, handle, protocol=pickle.HIGHEST_PROTOCOL)



k_o = io.loadmat('../../data/kernels/test_kernels.mat')
k = k_o['kernels']  

k_new = np.zeros((2000,9,9), dtype=np.float32)
for i in range(2000):
    k_new[i,:,:] = k[15:24,15:24,i]
k = k_new
with open('../../data/kernels/test_kernels.pickle', 'wb') as handle:
    pickle.dump(k, handle, protocol=pickle.HIGHEST_PROTOCOL)






