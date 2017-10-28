#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 2017
@author: Inom Mirzaev
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


def make_plots(img, segm, segm_pred):
    n_cols=3
    n_rows = len(img)

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[n_cols*mm])
        ax.imshow(img[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('MRI image', fontsize=20)
        ax = fig.add_subplot(gs[n_cols*mm+1])
        ax.imshow(segm[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('True Mask', fontsize=20)

        ax = fig.add_subplot(gs[n_cols*mm+2])
        ax.imshow(segm_pred[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('Predicted Mask', fontsize=20)
    return fig


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
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X_test[img_list].reshape(-1,img_rows, img_cols)
    segm = y_test[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    fig = make_plots(img, segm, segm_pred)
    fig.savefig('../images/best_predictions.png', bbox_inches='tight', dpi=300 )


    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X_test[img_list].reshape(-1,img_rows, img_cols)
    segm = y_test[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    fig = make_plots(img, segm, segm_pred)
    fig.savefig('../images/worst_predictions.png', bbox_inches='tight', dpi=300 )

if __name__=='__main__':
    check_predictions( )
