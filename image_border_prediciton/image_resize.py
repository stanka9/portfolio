"""
Author: Stanislava Poizlova
Matr.Nr.: K12023677
Exercise 5
"""

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

img = Image.open('//LAPTOP-3OOKEOOK/Users/marti/Documents/Stanu3ka/example_project/dataset/dataset/003/0000000.jpg')
img = ToTensor()(img)
out = img.resize((90,90)) #The resize operation on tensor.
ToPILImage()(out).save('test.png', mode='png')