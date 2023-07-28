"""
Author: Stanislava Poizlova
Matr.Nr.: K12023677
Exercise 5
"""
import torch
import pickle
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

# load model
model = torch.load('best_model.pt')
model.eval()

with open('stacked_arrays.pkl', 'rb') as d:
    dataset = pickle.load(d)

with open('input_arrays.pkl', 'rb') as d:
    input = pickle.load(d)


#print(dataset[0].shape)  # (2,90,90)

predictions = []

for i in dataset:
    i = torch.tensor(i, dtype=torch.float32)
    # print(i.shape) #(2,90,90)
    i = i.unsqueeze(0)
    # print(i.shape)  # (1,2,90,90)
    prediction = model(i)
    # print(prediction)
    prediction = prediction.squeeze(0)
    prediction = prediction.squeeze(0)
    #print(prediction.shape)
    predictions.append(prediction.detach().numpy())

#print(type(predictions[0].shape))

pred_denorm=[]
for i,j in zip(input,predictions):
    mean = i.mean()
    std = i.std()
    j[:] *= std
    j[:] += mean
    #print(j)
    #print(i)
    pred_denorm.append(j)

#print(input[0])

#load borders
with open('border_x.pkl', 'rb') as d:
    border_x = pickle.load(d)

with open('border_y.pkl', 'rb') as d:
    border_y = pickle.load(d)


prediction_array=[]

for i,x,y in zip(pred_denorm, border_x, border_y):

    known_array = np.zeros_like(i)
    known_array[x[0]:-x[1], y[0]:-y[1]] = 1
    target_array = i[known_array == 0]
    #print(target_array)
    prediction_array.append(target_array)


final=[]
for i in prediction_array:
    pred = np.asarray(i, dtype=np.uint8)
    final.append(pred)


#print(prediction_array[0].shape)

with open('example_targets.pkl', 'rb') as d:
    tar= pickle.load(d)

#print(prediction_array[0])
print(tar[0])
print(prediction_array[0])
#print(border_x[0])
#print(border_y[0])

print(type(prediction_array[0]))

with open('prediction_file.pkl', 'wb') as pfh:
    pickle.dump(final, pfh)