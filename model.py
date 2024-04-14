import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob as gb
from tqdm import tqdm
import pandas as pd
from skimage.io import imread

X_train = []
y_train = []
X_test = []

features_dir = 'augmented_features/'
targets_dir = 'augmented_targets/'
test_dir = 'test_directory/'

for file in tqdm(os.listdir(features_dir)):
    image = cv.imread(features_dir + file)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    X_train.append(image)

for file in tqdm(os.listdir(targets_dir)):
    image = cv.imread(targets_dir + file)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    y_train.append(image)

print(len(X_train))
print(len(y_train))

X_train = np.array(X_train)
y_train = np.array(y_train)


for file in tqdm(os.listdir(test_dir)):
    image = cv.imread(test_dir + file)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224), interpolation = cv.INTER_LINEAR)
    X_test.append(image)

print(len(X_test))

X_test = np.array(X_test)

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),8))) :
    plt.subplot(4,4,n+1)
    plt.imshow(X_train[i])
    plt.title(i)
    plt.subplot(4,4,8-n)
    plt.imshow(y_train[i],cmap='gray')
    plt.title(i)
    plt.axis('off')

plt.show()

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),16))) :
    plt.subplot(4,4,n+1)
    plt.imshow(X_test[i])
    plt.axis('off')

plt.show()

from sklearn.model_selection import train_test_split

y_train = np.expand_dims(y_train, axis = 3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.15, random_state = 42)


print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)

import sm_script as sm
# from segmentation_models import Unet
# from segmentation_models import get_preprocessing
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score

backbone = "resnet34"
preprocess_input = sm.get_preprocessing(backbone)

X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_valid)


model = sm.Unet(backbone, encoder_weights = "imagenet")
model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics = ['accuracy'])
print(model.summary())

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

EarlyStop = EarlyStopping(patience = 10, restore_best_weights = True)
Reduce_LR = ReduceLROnPlateau(monitor = 'val_accuracy', verbose = 2, factor = 0.5, min_lr = 0.00001)
model_check = ModelCheckpoint('model.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)
callback = [EarlyStop, Reduce_LR, model_check]


history = model.fit(X_train, y_train, batch_size = 16, epochs = 10, verbose = 1, validation_data =(X_valid, y_valid), callbacks=callback)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = "Training loss")
plt.plot(epochs, val_loss, 'r', label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('model.h5')

y_pred = model.predict(X_test)
plt.imshow(X_test[10])
plt.show()

plt.imshow(y_pred[10], cmap = 'gray')
plt.show()