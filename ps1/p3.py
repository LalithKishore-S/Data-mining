import pandas as pd
import numpy as np

data = pd.read_csv('/home/cslinux/Desktop/ps1/p3.csv')
print(data)

for i in range(len(data.columns)):
    print(f'Group{i + 1}')
    print(data[f'G{i+1}'])
    
    mean =  sum(data[f'G{i+1}'])/ data.shape[0]
    #print(data[f'G{i+1}'] - mean)
    print(f"Mean = {mean}")
    std = np.sqrt(sum((data[f'G{i+1}'] - mean)**2)/ data.shape[0])
    print(f"Std = {std}")
    