from keras.preprocessing.image import ImageDataGenerator
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

data_gen_args = dict(rotation_range = 180,
                    samplewise_std_normalization = False,
                    brightness_range = [0.1, 1],
                    width_shift_range = 0.5,
                    height_shift_range = 0.5,
                    shear_range = 50,
                    zoom_range = 0.1,
                    vertical_flip = True,
                    horizontal_flip = True,
                    fill_mode = 'reflect')

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    save_to_dir = 'None', target_size = (2*256,2*256), seed = 42):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
                        train_path,
                        #image_folder,
                        class_mode = None,
                        classes = [image_folder],
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        #save_to_dir = save_to_dir,
                        #save_prefix = image_save_prefix,
                        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
                        train_path,
                        #mask_folder,
                        class_mode = None,
                        classes = [mask_folder],
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        #save_to_dir = save_to_dir,
                        #save_prefix = mask_save_prefix,
                        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    #return train_generator
    #print(train_generator)
    #return train_generator

    for (img, mask) in train_generator:
        #print(img.shape)
        img_max = img.max()
        img /= img_max

        mask_max = mask.max()
        mask /= mask_max
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        #print(img, mask)
        yield (img, mask)
    

generator_instance = trainGenerator(1, 'data/', 'images', 'masks', data_gen_args, save_to_dir = 'None', target_size = (image_dimensions[0], image_dimensions[1]))

import segmentation_models as sm
# from segmentation_models import Unet
# from segmentation_models import get_preprocessing
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score

backbone = "resnet34"

preprocess_input = sm.get_preprocessing(backbone)

generator_instance = preprocess_input(generator_instance)

model = sm.Unet(backbone, encoder_weights = "imagenet")
model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics = ['accuracy'])
print(model.summary())

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

EarlyStop = EarlyStopping(patience = 10, restore_best_weights = True)
Reduce_LR = ReduceLROnPlateau(monitor = 'val_accuracy', verbose = 2, factor = 0.5, min_lr = 0.00001)
model_check = ModelCheckpoint('model_sm.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)
callback = [EarlyStop, Reduce_LR, model_check]

history = model.fit_generator(generator_instance, steps_per_epoch = 50, epochs = 20,
                  callbacks = callback)