import numpy as np
import pandas as pd

"""
num_samples = [10, 25, 30, 40]
maxi = np.max(num_samples)
data = []
for i in range(len(num_samples)):
    sample = np.random.randint(1000, size=(num_samples[i]))
    data.append(sample)

print(data)   
np.save('/home/cslinux/Desktop/ps1/p4.npy', np.array(data, dtype='object'))
"""

import matplotlib.pyplot as plt

array = np.load('/home/cslinux/Desktop/ps1/p4.npy', allow_pickle=True)
print(array.shape) 

for i in range(array.shape[0]):
    print(f'Attr{i+1}')
    mean = sum(array[i])/array[i].shape[0]
    std = np.sqrt(sum((array[i] - mean)**2)/ array[i].shape[0])
    print(f'Std = {std}')
    
def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out

_, ax = plt.subplots(1,4)
for i in range(array.shape[0]):
    y = pdf(array[i])
    ax[i].scatter(array[i], y)
    
plt.show()