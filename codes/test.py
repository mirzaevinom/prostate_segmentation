#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: mirzaev.1
"""

from __future__ import division, print_function
import numpy as np
print()
import matplotlib.pyplot as plt

from keras.optimizers import Adam

from models import actual_unet, simple_unet
from metrics import dice_coef, dice_coef_loss, numpy_dice
import os
import matplotlib.gridspec as gridspec

def check_predictions(n_best=3, n_worst=3 ):
    if not os.path.isdir('../images'):
        os.mkdir('../images')

    X_test = np.load('../data/test.npy')
    y_test = np.load('../data/test_masks.npy')

    img_rows = X_test.shape[1]
    img_cols = img_rows

    model = simple_unet(img_rows, img_cols)
    model.load_weights('../data/weights.h5')

    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
    score = model.evaluate(X_test, y_test, verbose=1)
    print()
    print('Test accuracy:', score[1])

    y_test = y_test.astype('float32')
    y_pred = model.predict( X_test, verbose=1)

    axis = tuple( range(1, y_test.ndim ) )
    scores = numpy_dice( y_test, y_pred , axis=axis)

    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y_test.sum(axis=axis) )[0]

    #Add some best and worst predictions
    img_list = []
    count = 0
    for ind in sort_ind:
        if count>n_best:
            break
        if ind in indice:
            img_list.append(ind)
            count+=1


    count = 0
    for ind in sort_ind[::-1]:

        if count>n_worst:
            break

        if ind in indice:
            img_list.append(ind)
            count+=1


    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X_test[img_list].reshape(-1,img_rows, img_cols)
    segm = y_test[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    n_cols=3
    n_rows = len(img_list)

    fig = plt.figure(figsize=[ 4*n_cols, int(4.3*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols, hspace=0.2 )

    for mm in range( len(img_list) ):

        ax = fig.add_subplot(gs[n_cols*mm])
        ax.imshow(img[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1

        ax = fig.add_subplot(gs[n_cols*mm+1])
        ax.imshow(segm[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1

        ax = fig.add_subplot(gs[n_cols*mm+2])
        ax.imshow(segm_pred[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1

    fig.savefig('../images/test_predictions.png', bbox_inches='tight', dpi=300 )

if __name__=='__main__':
    check_predictions( )
