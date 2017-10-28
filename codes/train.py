#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function
import numpy as np
print()

import dicom
from collections import defaultdict
import os, pickle, sys
import shutil
import matplotlib.pyplot as plt
import nrrd
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist


from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint


from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator


from models import actual_unet, simple_unet
from metrics import dice_coef, dice_coef_loss, numpy_dice

from augmenters import elastic_transform

def dicom_to_array(img_rows, img_cols):

    for direc in ['train', 'test', 'validate']:

        fname = '../data/' + direc + '_dict.pkl'
        PathDicom = '../data/' + direc + '/'
        dcm_dict = dict()  # create an empty list

        for dirName, subdirList, fileList in os.walk(PathDicom):

            if any(".dcm" in s for s in fileList):
                ptn_name = dirName.split('/')[3]
                fileList = filter(lambda x: '.dcm' in x, fileList)
                indice = [ int( fname[:-4] ) for fname in fileList]
                imgs = np.zeros( [indice[-1]+1, img_rows, img_cols])

                for filename in np.sort(fileList):

                    img = dicom.read_file(os.path.join(dirName,filename)).pixel_array.T
                    img = equalize_hist( img.astype(int) )
                    img = resize( img, (img_rows, img_cols), preserve_range=True)
                    imgs[int(filename[:-4])] = img

                dcm_dict[ptn_name] = imgs

        imgs = []
        img_masks = []

        for patient in dcm_dict.keys():
            for fnrrd in os.listdir(PathDicom):

                if fnrrd.startswith(patient) and fnrrd.endswith('nrrd'):
                    masks = np.rollaxis(nrrd.read(PathDicom + fnrrd)[0], 2)
                    rescaled = np.zeros( [ len(masks), img_rows, img_cols])
                    for mm in range(len(rescaled)):
                        rescaled[mm] = resize( masks[mm], (img_rows, img_cols), preserve_range=True)/2.0

                    masks = rescaled.copy()

                    #Check if the dimension of the masks and the images match
                    if len(dcm_dict[patient]) != len(masks) :
                        print('Dimension mismatch for', patient, 'in folder', direc)
                    else:
                        img_masks.append(masks)
                        imgs.append( dcm_dict[patient] )

                    break

        imgs = np.concatenate(imgs, axis=0).reshape(-1, img_rows, img_cols, 1)
        img_masks = np.concatenate(img_masks, axis=0).reshape(-1, img_rows, img_cols, 1)

        #I will do only binary classification for now
        img_masks = np.array(img_masks>0.45, dtype=int)
        np.save('../data/' + direc + '.npy', imgs)
        np.save('../data/' + direc + '_masks.npy', img_masks)


def load_data():

    X_train = np.load('../data/train.npy')
    y_train = np.load('../data/train_masks.npy')
    X_test = np.load('../data/test.npy')
    y_test = np.load('../data/test_masks.npy')
    X_val = np.load('../data/validate.npy')
    y_val = np.load('../data/validate_masks.npy')

    return X_train, y_train, X_test, y_test, X_val, y_val

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * drop**int((1 + epoch) / epochs_drop)
    return lrate

def keras_fit_generator(img_rows=96, img_cols=96, n_imgs=10**4, batch_size=32, regenerate=True):

    if regenerate:
        dicom_to_array(img_rows, img_cols)
        #preprocess_data()

    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    img_rows = X_train.shape[1]
    img_cols = img_rows

    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2)#,
        #preprocessing_function=elastic_transform)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train,seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    from itertools import izip
    train_generator = izip(image_generator, mask_generator)


    model = simple_unet( img_rows, img_cols)
    #model.load_weights('../data/weights.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint(
        '../data/weights.h5', monitor='val_loss', save_best_only=True)

    lrate = LearningRateScheduler(step_decay)

    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])

    model.fit_generator(
                        train_generator,
                        steps_per_epoch=n_imgs//batch_size,
                        epochs=30,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=[model_checkpoint, lrate],
                        use_multiprocessing=True)

    score = model.evaluate(X_test, y_test, verbose=2)

    print()
    print('Test accuracy:', score[1])

import time

start = time.time()
keras_fit_generator(img_rows=96, img_cols=96, regenerate=True,
                   n_imgs=15*10**4, batch_size=32)

end = time.time()

print('Elapsed time:', round((end-start)/60, 2 ) )
