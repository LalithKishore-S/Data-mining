import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.randint(low = 10, high = 85, size = (100,4))
#print(data)

for i in range(data.shape[1]):
    print(f'Col{i+1}')
    temp = data[:, i]
    temp = sorted(temp)
    n = len(temp)
    print(f'First quartile = {temp[round((n+1)/4) - 1]}')
    print(f'Second quartile = {temp[round((n+1)/2) - 1]}')
    print(f'Third quartile = {temp[round(3*(n+1)/4) - 1]}')
    
_, ax = plt.subplots(1,4)
for i in range(data.shape[1]):
    temp = data[:, i]
    ax[i].boxplot(temp)
    ax[i].set_title(f'Col{i+1}')
    
_, ax = plt.subplots(1,4)
for i in range(data.shape[1]):
    temp = data[:, i]
    ax[i].hist(temp, bins = 80,color = 'k')
    ax[i].set_title(f'Col{i+1}')