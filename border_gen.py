import os
import numpy as np
import matplotlib.pyplot as plt  # for visualising and debugging
from scipy.ndimage.morphology import distance_transform_edt
from skimage.io import imsave, imread
from skimage.segmentation import find_boundaries
#from demos.unet.file_sorter import get_user_args
from skimage import data, io, filters, measure
from pathlib import Path
import glob
from PIL import Image
import skimage.transform as trans

W_0, SIGMA = 25, 25


def construct_weights_and_mask(img):
    img = measure.label(img)
    seg_boundaries = find_boundaries(img, mode='inner')

    bin_img = img > 0
    # take segmentations, ignore boundaries
    binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

    foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
    background_weight = 1 - foreground_weight

    # build euclidean distances maps for each cell:
    cell_ids = [x for x in np.unique(img) if x > 0]
    distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))

    for i, cell_id in enumerate(cell_ids):
        distances[..., i] = distance_transform_edt(img != cell_id)

    # we need to look at the two smallest distances
    distances.sort(axis=-1)
    weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[...,0] + distances[...,1]) ** 2))
    weight_map[binary_with_borders] = foreground_weight
    weight_map[~binary_with_borders] += background_weight

    return weight_map

import cv2 as cv
import sys

path = os.getcwd()
print(path)
path_label = path + '/data/masks/'
path_border = path + '/data/borders/'
path_image = path + '/data/images/'
#%%
counter = 0
print(len(glob.glob(path_label+"*.png",recursive=True)))
print(path_label)

#%%
for file in glob.glob(path_label+"*.png",recursive=True):

        img = imread(file, as_gray = True)
#         img = trans.resize(img,(image_dimensions[0],image_dimensions[1]))
        # a = img.max()
        # img[img < 0.85*a] = 0
        # img[img>=0.85*a] = 255
        # print(img.shape)

        print(os.path.join(path_border, os.path.basename(file)))
        #sys.exit()
        # Image.fromarray(img).save(path_label + os.path.basename(file))

        
        temp = construct_weights_and_mask(img)
        cv.imwrite(path_border+os.path.basename(file), temp)
        #Image.fromarray(temp).save(path_border + os.path.basename(file))


        counter += 1

print("done")
