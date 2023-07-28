"""
Author: Stanislava Poizlova
Matr.Nr.: K12023677
Exercise 4
"""

import numpy as np
import copy


def ex4(image_array, border_x, border_y):
    original_array = copy.deepcopy(image_array)

    # ImplementationError
    if isinstance(image_array, np.ndarray):
        pass
    else:
        raise NotImplementedError("image_array is not a numpy array")

    if isinstance(image_array, np.ndarray) and image_array.ndim != 2:
        raise NotImplementedError("image_array is not a 2D array")

    # ValueError1-conversion
    try:
        border_x = int(border_x[0]), int(border_x[1])
        border_y = int(border_y[0]), int(border_y[1])

    except ValueError as error:
        raise error

    # ValueError-pixels
    remaining_x = image_array.shape[0] - border_x[0] - border_x[1]
    remaining_y = image_array.shape[1] - border_y[0] - border_y[1]

    if remaining_x < 16 or remaining_y < 16:
        raise ValueError("shape of the remaining known image pixels would be smaller than (16, 16)")

    # ValueError-less than one
    if border_x[0] < 1:
        raise ValueError
    elif border_y[0] < 1:
        raise ValueError
    elif border_x[1] < 1:
        raise ValueError
    elif border_y[1] < 1:
        raise ValueError

    #input_array
    input_array = np.full_like(image_array, image_array, dtype=image_array.dtype)
    input_array[:, :border_y[0]] = 0
    input_array[:border_x[0], :] = 0
    input_array[:, input_array.shape[1] - border_y[1]:] = 0
    input_array[input_array.shape[0] - border_x[1]:, :] = 0

    #known_array
    known_array = np.zeros_like(image_array)
    known_array[border_x[0]:-border_x[1], border_y[0]:-border_y[1]] = 1

    #target_array
    target_array = image_array[known_array == 0]

    return original_array, input_array, known_array, target_array

'''''''''''''''''''''''''''''''''''
import os
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
input_path = "01.jpg"
image_files = sorted(glob.glob(os.path.join(input_path),
                               recursive=True))


image = Image.open(image_files[0])  # This returns a PIL image
image = np.array(image) #convertin the first image to numpy array

print(ex4(image, (1,2), (3,2)))
'''''''''''''''''''''''''''''''''''''''''''''''''''''