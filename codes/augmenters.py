
from __future__ import division, print_function
import numpy as np


from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from keras.preprocessing.image import apply_transform, transform_matrix_offset_center

from keras.preprocessing.image import ImageDataGenerator


def elastic_transform(image, alpha=0.0, sigma=0.25, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Based on: https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def random_rotation(x,
                    y,
                    rg=15,
                    row_index=0,
                    col_index=1,
                    channel_index=2,
                    fill_mode='nearest',
                    cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def augment_data(N=2):

    X_train = np.load('../data/train.npy')
    y_train = np.load('../data/train_masks.npy')

    res_image = np.zeros_like(X_train)
    res_mask = np.zeros_like(y_train)
    rot_imgs = np.zeros_like(X_train)
    rot_masks = np.zeros_like(y_train)

    for nn in range(N):
        for mm in range(len(res_image)):

            res_image[mm, :, :, 0], res_mask[mm, :, :, 0] = elastic_transform(
                X_train[mm, :, :, 0], y_train[mm, :, :, 0], 100, 20)
            rot_imgs[mm], rot_masks[mm] = random_rotation(
                X_train[mm], y_train[mm])

        X_train = np.concatenate([X_train, res_image, rot_imgs], axis=0)
        y_train = np.concatenate([y_train, res_mask, rot_masks], axis=0)

    np.save('../data/train.npy', X_train)
    np.save('../data/train_masks.npy', y_train)

def keras_augment_data(N=10000, batch_size=1000):

    X_train = np.load('../data/train.npy')
    y_train = np.load('../data/train_masks.npy')

    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    from itertools import izip
    train_generator = izip(image_generator, mask_generator)

    n_aug = 0

    for X_batch, y_batch in train_generator:

        X_train = np.concatenate([X_train, X_batch], axis=0)
        y_train = np.concatenate([y_train, y_batch], axis=0)
        n_aug+=batch_size
        if n_aug>N:
            break

    np.save('../data/train.npy', X_train)
    np.save('../data/train_masks.npy', y_train)
