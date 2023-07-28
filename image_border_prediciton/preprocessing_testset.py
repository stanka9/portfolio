"""
Author: Stanislava Poizlova
Matr.Nr.: K12023677
Exercise 5
"""
import pickle
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

with open('example_testset.pkl', 'rb') as f:
    dataset = pickle.load(f)



#get list of available keys
keys = ['input_arrays','known_arrays','sample_ids','borders_x','borders_y']

#get values for given key
dataset_len=dataset['input_arrays']
print(len(dataset_len))

#collect borders
border_x = []
border_y = []

for i in range(len(dataset_len)):
    d=dataset[keys[3]][i]
    d=d.tolist()
    border_x.append(d)

for i in range(len(dataset_len)):
    d=dataset[keys[4]][i]
    d = d.tolist()
    border_y.append(d)

#collect input arrays
input_arrays = []

for i in range(len(dataset_len)):
    d=dataset[keys[0]][i]
    d = np.asarray(d, dtype=np.float32)
    input_arrays.append(d)

print(input_arrays[0])

with open('input_arrays.pkl', 'wb') as f:
    pickle.dump(input_arrays, f)

#collect known_arrays
known_arrays = []

for i in range(len(dataset_len)):
    d=dataset[keys[1]][i]
    d = np.asarray(d, dtype=np.float32)
    known_arrays.append(d)


#sample ids
sample_ids = []

for i in range(len(dataset_len)):
    d=dataset[keys[2]][i]
    sample_ids.append(d)


#I have list for all the arrays needed
input_array_norm=[]

for array in input_arrays:
    mean = array.mean()
    std = array.std()
    array[:] -= mean
    array[:] /= std
    input_array_norm.append(array)

stacked_arrays=[]


for i,j in zip(known_arrays,input_arrays):
    stacked_array = np.stack((i, j))
    stacked_arrays.append(stacked_array)
    #print(input_arrays)
    #print(stacked_arrays[0].shape)

print(len(stacked_arrays))

#print(input_arrays[0])
#I have normalized stacked_arrays

with open('stacked_arrays.pkl', 'wb') as f:
    pickle.dump(stacked_arrays, f)


with open('border_x.pkl', 'wb') as f:
    pickle.dump(border_x, f)

with open('border_y.pkl', 'wb') as f:
    pickle.dump(border_y, f)

