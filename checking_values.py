import cv2 as cv
import numpy as np

image = cv.imread('masks/mask8.png', cv.IMREAD_UNCHANGED)

if image is None:
    print('Scream')

print(type(image))

uniques, counts = np.unique(image, return_counts=True)
print("Unique values in image array: ", uniques)
print("Number of occurrences: ", counts)

if image[0][0] == 0:
    print("First pixel in image is black.")
else:
    print("First pixel in image is white.")