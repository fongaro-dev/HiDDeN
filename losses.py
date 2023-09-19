import tensorflow as tf
from tensorflow import _keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Conv2DTranspose, BatchNormalization, ReLU, Concatenate

def rgb_diff(original_img, encoded_img):
    MRSE = tf.keras.losses.MeanSquaredError(
        reduction='auto',
        name='mean_squared_error'
    )
    return MRSE(original_img, encoded_img)
