"""
Author: Stanislava Poizlova
Matr.Nr.: K12023677
Exercise 5
"""
import numpy as np
from torch.utils.data import Dataset
import pickle
from ex4 import ex4
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

random.seed(5)


class Images(Dataset):

    def __init__(self):
        # Get dataset
        with open('resized_new', 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image_array = self.dataset[index]

        # create random possible tuples for borders
        possible = [5, 6, 7, 8, 9, 10]
        possible_tuples = []

        for i in possible:
            for j in possible:
                if i + j <= 15:
                    possible_tuples.append((i, j))
                else:
                    pass

        # apply ex4 with random borders
        original_array, input_array, known_array, target_array = ex4(image_array,
                                                                     border_x=random.choice(possible_tuples),
                                                                     border_y=random.choice(possible_tuples))
        # covert to float32
        input_array = np.asarray(input_array, dtype=np.float32)
        original_array = np.asarray(original_array, dtype=np.float32)
        known_array = np.asarray(known_array, dtype=np.float32)

        # get mean and std from input layer
        mean = input_array.mean()
        std = input_array.std()

        # normalize input layer
        input_array[:] -= mean
        input_array[:] /= std

        # normalize original array
        original_array[:] -= mean
        original_array[:] /= std

        '''''''''
        not sure if mean and std should not be used the same for input and original layer
        '''
        # create stacked array
        stacked_array = np.stack((known_array, input_array))

        original_array= TF.to_tensor(original_array)

        #original_array = original_array.unsqueeze(0)

        #print(original_array.shape)
        #print(stacked_array.shape)
        # original array=target, stacked array=input
        return original_array, stacked_array, index


x = random.sample(range(1, 30000), 5)
print(x)

dataset = Images()

trainingset = torch.utils.data.Subset(dataset, indices=x)
trainloader = torch.utils.data.DataLoader(trainingset, batch_size=4, shuffle=True, num_workers=0)

torch.set_printoptions(edgeitems=8)

for data in trainloader:
    original_array, stacked_array, index = data
    #print(stacked_array)
