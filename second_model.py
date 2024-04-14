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
#from keras.models import load_model
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dropout, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard 
import tensorflow as tf

def unet(input_size = (224,224,3)):
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet(input_size=(224,224,3))

print(model.summary())
optimizer_adam=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.99)
EarlyStop=EarlyStopping(patience=10,restore_best_weights=True)
Reduce_LR=ReduceLROnPlateau(monitor='val_accuracy',verbose=2,factor=0.5,min_lr=0.00001)
model_check=ModelCheckpoint('model.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
tensorbord=TensorBoard(log_dir='logs')
callback=[EarlyStop , Reduce_LR,model_check,tensorbord]

model.compile(optimizer=optimizer_adam,loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=10,batch_size=16,
                  callbacks=callback, verbose=1)

y_pred=model.predict(X_test)
y_pred=(y_pred > 0.3).astype(np.uint8)

plt.imshow(X_test[10])
plt.show()

plt.imshow(y_pred[10],cmap='gray')
plt.show()

