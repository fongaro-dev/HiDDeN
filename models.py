import tensorflow as tf
from tensorflow import _keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Conv2DTranspose, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model

def conv_bn_relu_block(input_tensor, filters, kernel_size, strides=(1, 1), padding='same'):
    """
    Create a Conv-BN-ReLU block in TensorFlow/Keras.

    Args:
        input_tensor: Input tensor to the block.
        filters: Number of filters (output channels) for the Conv2D layer.
        kernel_size: Size of the convolutional kernel, e.g., (3, 3).
        strides: Strides for the convolution (default is (1, 1)).
        padding: Padding type, e.g., 'same' or 'valid' (default is 'same').

    Returns:
        Output tensor of the Conv-BN-ReLU block.
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def create_encoder_decoder_model(input_shape):
    # Define the input layers for both images
    cover_img = Input(shape=input_shape)
    hidden_img = Input(shape=input_shape)

    # Apply conv-BN-Relu to original image (5 blocks in original paper)
    conv_relu1_cover = conv_bn_relu_block(cover_img, filters=64, kernel_size=(3, 3))
    conv_relu2_cover = conv_bn_relu_block(conv_relu1_cover, filters=64, kernel_size=(3, 3))
    conv_relu3_cover = conv_bn_relu_block(conv_relu2_cover, filters=64, kernel_size=(3, 3))
    # conv_relu4_cover = conv_bn_relu_block(conv_relu3_cover, filters=64, kernel_size=(3, 3))
    # conv_relu5_cover = conv_bn_relu_block(conv_relu4_cover, filters=64, kernel_size=(3, 3))

    # Apply conv-BN-Relu to hidden image
    # (the orginal message is just repeated HxW times and concatenated in the original paper)
    conv_relu1_hidden = conv_bn_relu_block(hidden_img, filters=64, kernel_size=(3, 3))
    conv_relu2_hidden = conv_bn_relu_block(conv_relu1_hidden, filters=64, kernel_size=(3, 3))
    conv_relu3_hidden = conv_bn_relu_block(conv_relu2_hidden, filters=64, kernel_size=(3, 3))

    # Skip layer: Concatenate the conv-BN-relu results
    concat1 = Concatenate()([conv_relu3_cover, conv_relu3_hidden, cover_img, hidden_img])
    conv_relu1_concat = conv_bn_relu_block(concat1, filters=64, kernel_size=(3, 3))
    encoded_img = Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), padding='same')(conv_relu1_concat)

    # Add gaussian noise
    encoded_img_wnoise = gaussian_noise_layer(encoded_img, 0.02)

    # pool1 = MaxPooling2D((2, 2))(conv1)

    # Apply conv-BN-Relu to encoded image (7 blocks in original paper)
    conv_relu1 = conv_bn_relu_block(encoded_img_wnoise, filters=64, kernel_size=(3, 3))
    conv_relu2 = conv_bn_relu_block(conv_relu1, filters=64, kernel_size=(3, 3))
    conv_relu3 = conv_bn_relu_block(conv_relu2, filters=64, kernel_size=(3, 3))
    conv_relu4 = conv_bn_relu_block(conv_relu3, filters=64, kernel_size=(3, 3))
    conv_relu5 = conv_bn_relu_block(conv_relu4, filters=64, kernel_size=(3, 3))
    # conv_relu6 = conv_bn_relu_block(conv_relu5, filters=64, kernel_size=(3, 3))
    # conv_relu7 = conv_bn_relu_block(conv_relu6, filters=64, kernel_size=(3, 3))


    # Skip layer: Concatenate the conv-BN-relu results
    concat1 = Concatenate()([conv_relu5, encoded_img_wnoise])
    conv_relu1_concat = conv_bn_relu_block(concat1, filters=64, kernel_size=(3, 3))
    decoded = Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), padding='same')(conv_relu1_concat)

    # pool1 = MaxPooling2D((2, 2))(conv1)

    # Create the model
    model = Model(inputs=[cover_img, hidden_img], outputs=[encoded_img, decoded])

    return model

