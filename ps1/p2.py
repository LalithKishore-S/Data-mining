import pandas as pd
import numpy as np

data = pd.read_csv('/home/cslinux/Desktop/ps1/p2.csv')
print(data)
data['Grade'] = data['Grade'].astype('str')
print(data.dtypes)

lower_lim = []
upper_lim = []
mid = []
cf = []
for i in range(data.shape[0]):
    t = data['Grade'].iloc[i].split('-')
    lower_lim.append(int(t[0]) -0.5)
    upper_lim.append(int(t[1]) +0.5)
    mid.append((upper_lim[-1] + lower_lim[-1])/2)
    if len(cf) >= 1:
        cf.append(cf[-1] + data['Frequency'].iloc[i])
    else:
        cf.append(data['Frequency'].iloc[i])

data['lower_lim'] = lower_lim
data['upper_lim'] = upper_lim
data['mid'] = mid
data['cf'] = cf

print(data)

print(f"Mean = {sum(data['mid'] * data['Frequency'])/ sum(data['Frequency'])}")

median_class = -1
cum_freq = sum(data['Frequency'])
for i in range(data.shape[0]):
    if data['cf'].iloc[i] >= cum_freq/2:
        median_class = i
        break
    
print(f'Median = {data['lower_lim'].iloc[median_class] + 10 *  (cum_freq/2 - data['cf'].iloc[median_class - 1 ])/ data['Frequency'].iloc[median_class]}')


modal_class = 0
modal_class_freq = data['Frequency'].iloc[0]
for i in range(1, data.shape[0]):
    if data['Frequency'].iloc[i] > modal_class_freq:
        modal_class = i
        modal_class_freq = data['Frequency'].iloc[i]
        
l = data['lower_lim'].iloc[modal_class]
f1 = data['Frequency'].iloc[modal_class]
f0 = data['Frequency'].iloc[modal_class]
f2 = data['Frequency'].iloc[modal_class + 1]

print(f"Mode = {l + (f1-f0)/(2*f1 - f0 - f2) * 10}")