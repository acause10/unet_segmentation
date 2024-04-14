from keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

model = load_model('unet_cell_segment_model3.hdf5')
image_pred = cv.imread('test_directory/1.tif', cv.IMREAD_GRAYSCALE)
image_pred = cv.resize(image_pred, dsize = (512,512), interpolation = cv.INTER_LINEAR)

plt.imshow(image_pred, alpha = 0.5)
image_pred = np.expand_dims(image_pred, axis = 0)
print(image_pred.shape)

y_pred = model.predict(image_pred)
y_pred = y_pred.reshape(512,512)
plt.imshow(y_pred, cmap = 'gray')
plt.show()