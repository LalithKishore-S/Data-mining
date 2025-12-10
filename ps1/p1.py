import numpy as np


samples = np.random.randint(low = 0, high = 100, size = (10,))

mean = 0
for i in range(samples.shape[0]):
    mean += samples[i]
mean /= samples.shape[0]

print(f"Mean = {mean}")

sorted_samples = np.array(sorted(samples))
if sorted_samples.shape[0] %2 == 0:
    median = (sorted_samples[sorted_samples.shape[0]//2 - 1] + sorted_samples[sorted_samples.shape[0]//2])/2
else:
    median = (sorted_samples[(sorted_samples.shape[0] + 1)//2 - 1])
    

print(sorted_samples)
print(f'Median = {median}')

dictionary = {}
maxi = 0 
for i in range(samples.shape[0]):
    if sorted_samples[i] in dictionary.keys():
        dictionary[sorted_samples[i]] += 1
    else:
        dictionary[sorted_samples[i]] = 1
    if dictionary[sorted_samples[i]] > maxi:
        maxi = dictionary[sorted_samples[i]]
        
print("Mode : ")
for i in dictionary.keys():
    if dictionary[i] == maxi:
        print(i)
        


