from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D


def actual_unet( img_rows, img_cols, N = 2):
    """This model is based on:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(
        2**(N + 3), (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(
        2**(N + 3), (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        2**(N + 4), (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(
        2**(N + 4), (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate(
        [
            Conv2DTranspose(
                2**(N + 3),
                (2, 2), strides=(2, 2), padding='same')(conv5), conv4
        ],
        axis=3)
    conv6 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(
        2**(N + 3), (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate(
        [
            Conv2DTranspose(
                2**(N + 2),
                (2, 2), strides=(2, 2), padding='same')(conv6), conv3
        ],
        axis=3)
    conv7 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate(
        [
            Conv2DTranspose(
                2**(N + 1),
                (2, 2), strides=(2, 2), padding='same')(conv7), conv2
        ],
        axis=3)
    conv8 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate(
        [
            Conv2DTranspose(2**N, (2, 2), strides=(2, 2),
                            padding='same')(conv8), conv1
        ],
        axis=3)
    conv9 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def simple_unet( img_rows, img_cols, N = 3):

    print(img_rows, img_cols)

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(
        2**(N + 2), (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate(
        [
            Conv2D(2**(N+1), 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv3)),
            conv2
        ],

        axis=3)
    conv4 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(up1)
    conv4 = Conv2D(
        2**(N + 1), (3, 3), activation='relu', padding='same')(conv4)


    up2 = concatenate(
        [
         Conv2D(2**(N), 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv4)),
         conv1
        ],
        axis=3)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(up2)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(conv5)

    conv6 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])

    return model
