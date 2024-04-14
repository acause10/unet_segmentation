from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import cv2 as cv

datagen = ImageDataGenerator(
    rotation_range=179,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect")

# #Read image
# single_image = io.imread('Images_to_segment_cherry_pick/A_1,6_ch00.tif')
# single_image = cv.cvtColor(single_image, cv.COLOR_GRAY2RGB)

# # We need 1,1024,1024,3 shape
# single_image = single_image.reshape((1, ) + single_image.shape )

dataset_targets = []
dataset_features = []

import numpy as np
import os
from PIL import Image

targets_image_directory = "Segmented_images/"
features_image_directory = "Images_to_segment_cherry_pick/"

size = 224
target_images = os.listdir(targets_image_directory)
feature_images = os.listdir(features_image_directory)
#print(target_images) prints out segmented images
#print(feature_images)
#print(target_images)

for i, image_name in enumerate(target_images):
    if (image_name.split('.')[1] == 'jpg'): #image_name.split('.')[1] je sufix slike
        image = io.imread(targets_image_directory + image_name)
        if image is None:
            print("Reading unsuccessful!")
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        if image is None:
            print("Conversion unsuccessful!")
        image = cv.resize(image, (size, size), interpolation = cv.INTER_LINEAR)
        if image is None:
            print("Resizing unsuccessful!")
        dataset_targets.append(np.array(image))

loaded_targets = np.array(dataset_targets)
#print(loaded_targets)

for batch in datagen.flow(loaded_targets, batch_size = 16, \
    save_to_dir = "augmented_targets", save_prefix = 'target', save_format = 'jpg', seed = 42):
    i += 1
    if i > 125:
        break

# # the same we do for our feature images
for i, image_name in enumerate(feature_images):
    if (image_name.split('.')[1] == 'tif'): #image_name.split('.')[1] je sufix slike
        image = io.imread(features_image_directory + image_name)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        image = cv.resize(image, (size, size), interpolation = cv.INTER_LINEAR)
        dataset_features.append(np.array(image))

loaded_features = np.array(dataset_features)

for batch in datagen.flow(loaded_features, batch_size = 16, \
    save_to_dir = "augmented_features", save_prefix = 'feature', save_format = 'jpg', seed = 42):
    
    i += 1
    if i > 125:
        break
