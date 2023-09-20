import tensorflow as tf
from tensorflow import _keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Conv2DTranspose, BatchNormalization, ReLU, Concatenate

def MSE(y_true, y_pred):
  """ return the per sample mean squared error """
  outputs = tf.square(y_true - y_pred)
  return tf.reduce_mean(outputs, axis = list(range(1, len(y_true.shape))))

def rgb_diff(original_img, encoded_img):
    return MSE(original_img, encoded_img)
