from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt

image_dimensions = (512, 512, 1)

# train_datagen = ImageDataGenerator(rescale = 1/255,
#                                 rotation_range = 180,
#                                 brightness_range = [0.1, 1],
#                                 width_shift_range = 0.5,
#                                 height_shift_range = 0.5,
#                                 shear_range = 50,
#                                 vertical_flip = True,
#                                 horizontal_flip = True,
#                                 fill_mode = 'reflect')

# train_generator = train_datagen.flow_from_directory(
#                             'Images_to_segment_cherry_pick',
#                             target_size = (256, 256),
#                             batch_size = 16,
# )

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


data_gen_args = dict(rotation_range = 180,
                    samplewise_std_normalization = False,
                    #brightness_range = [0.1, 1],
                    width_shift_range = 0.5,
                    height_shift_range = 0.5,
                    shear_range = 50,
                    zoom_range = 0.1,
                    vertical_flip = True,
                    horizontal_flip = True,
                    fill_mode = 'reflect')

def trainGenerator(batch_size, train_path, image_folder, mask_folder, border_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    save_to_dir = 'None', target_size = (2*256,2*256), seed = 42):

    image_datagen = ImageDataGenerator(**aug_dict)
    border_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
                        train_path,
                        #image_folder,
                        class_mode = None,
                        classes = [image_folder],
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        seed = seed)

    border_generator = border_datagen.flow_from_directory(
                        train_path,
                        classes = [border_folder],
                        class_mode = None,
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
                        train_path,
                        #mask_folder,
                        class_mode = None,
                        classes = [mask_folder],
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        seed = seed)
    
    train_generator = zip(image_generator, border_generator, mask_generator)
    #return train_generator
    #print(train_generator)
    #return train_generator

    for (img, border, mask) in train_generator:
        #print(img.shape)
        img_max = img.max()
        img /= img_max

        mask_max = mask.max()
        mask /= mask_max
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        #print(img, mask)

        border /= border.max()
        border *= 5
        border +=1

        stacked = np.stack((border,mask), axis=3)
        stacked = np.reshape(stacked, (stacked.shape[0],stacked.shape[1],stacked.shape[2],stacked.shape[3]))

        yield (img, stacked)
    

generator_instance = trainGenerator(1, 'data/', 'images', 'masks', 'borders', data_gen_args, save_to_dir = 'None', target_size = (image_dimensions[0], image_dimensions[1]))

###############################################################
# Model build - simple
import tensorflow as tf
def unet(input_size = (512, 512, 1)):

    inputs = tf.keras.layers.Input(input_size)
    #s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
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


model = unet(input_size = image_dimensions)
#print(model.summary()) 

############################################################################
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

optimizer_adam = Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.99)
EarlyStop = EarlyStopping(patience = 10, restore_best_weights = True)
Reduce_LR = ReduceLROnPlateau(monitor = 'loss', verbose = 2, factor = 0.5, min_lr = 0.00001)
model_check = ModelCheckpoint('unet_cell_segment_last_model.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)

callback = [EarlyStop, Reduce_LR, model_check]

model.compile(optimizer = optimizer_adam, loss = weighted_binary_cross_entropy, metrics = ['mae'])
##############################################################################

history = model.fit_generator(generator_instance, steps_per_epoch = 50, epochs = 20,
                  callbacks = callback)