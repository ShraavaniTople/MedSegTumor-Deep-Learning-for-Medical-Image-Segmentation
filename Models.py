# Medical_Image_Segmentation.ipynb
# (Jupyter notebook for building and training the U-Net model)
# Code can be written in a Jupyter notebook

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Decoder
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    up1 = UpSampling2D(size=(2, 2))(conv2)
    up1 = concatenate([conv1, up1], axis=3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(up1)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
input_shape = (256, 256, 3)  # Adjust according to your image size and channels
model = unet_model(input_shape)
model.summary()
