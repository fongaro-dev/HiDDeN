import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import _keras

from datetime import datetime
from packaging import version
from dataset_utils import split_dataset

import numpy as np
import os
import os, stat
import shutil

from models import create_encoder_model, create_decoder_model
from losses import rgb_diff

from IPython import display

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Clear out any prior log data.
shutil.rmtree("logs", onerror=remove_readonly)

# Sets up a timestamped log directory.
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
encoder_log_dir = 'logs/gradient_tape/' + current_time + '/encoder'
decoder_log_dir = 'logs/gradient_tape/' + current_time + '/decoder'
encoder_summary_writer = tf.summary.create_file_writer(encoder_log_dir)
decoder_summary_writer = tf.summary.create_file_writer(decoder_log_dir)

# Sets up a timestamped log directory.
logdir = "logs/train_data/" + current_time
file_writer = tf.summary.create_file_writer(logdir)



split0, split1 = tfds.even_splits('all', n=2)
ds_cover = tfds.load('stl10', split=split0 , shuffle_files=True)
ds_hidden = tfds.load('stl10', split=split1 , shuffle_files=True)


[ds_cover_train,  ds_cover_val , ds_cover_test] = split_dataset(ds_cover)
[ds_hidden_train, ds_hidden_val, ds_hidden_test] = split_dataset(ds_hidden)


print(f"Length of clear train set: {len(ds_cover_train)}, validation: {len(ds_cover_val)}, and test: {len(ds_cover_test)}")
print(f"Length of hidden train set: {len(ds_hidden_train)}, validation: {len(ds_hidden_val)}, and test: {len(ds_hidden_test)}")

# train_data = ds_cover_train.load_data()

tensor_img_batch = list(ds_cover_train)[0]['image']

# with file_writer.as_default():
#   # Don't forget to reshape.
#   images = np.reshape(tensor_img_batch[0:25], (-1, 96, 96, 3))
#   tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

def normalize(ds):
    # print(image)
    image_ds = (tf.cast(ds['image'],tf.float32)) / 255.0
    return image_ds

ds_cover_train = ds_cover_train.map(normalize)
ds_cover_val = ds_cover_val.map(normalize)
ds_cover_test = ds_cover_test.map(normalize)
ds_hidden_train = ds_hidden_train.map(normalize)
ds_hidden_val = ds_hidden_val.map(normalize)
ds_hidden_test = ds_hidden_test.map(normalize)

# Example usage:
input_shape = (96, 96, 3)  # Adjust input size as needed
encoder_model = create_encoder_model(input_shape)
decoder_model = create_decoder_model(input_shape)

encoder_model.summary()
decoder_model.summary()

encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
decoder_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder_optimizer=encoder_optimizer,
                                 decoder_optimizer=decoder_optimizer,
                                 encoder_model=encoder_model,
                                 decoder_model=decoder_model)

train_loss_enc = tf.keras.metrics.Mean('train_loss_enc', dtype=tf.float32)
train_loss_dec = tf.keras.metrics.Mean('train_loss_dec', dtype=tf.float32)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(cover_imgs, hidden_imgs):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
      encoded_imgs = encoder_model([cover_imgs, hidden_imgs], training=True)
      decoded_imgs = decoder_model(encoded_imgs, training=True)

      enc_loss = rgb_diff(encoded_imgs, cover_imgs)
      dec_loss = rgb_diff(decoded_imgs, hidden_imgs)

    train_loss_enc(enc_loss)
    train_loss_dec(dec_loss)

    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder_model.trainable_variables)
    gradients_of_decoder = dec_tape.gradient(dec_loss, decoder_model.trainable_variables)

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder_model.trainable_variables))
    decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder_model.trainable_variables))


val_loss_enc = tf.keras.metrics.Mean('val_loss_enc', dtype=tf.float32)
val_loss_dec = tf.keras.metrics.Mean('val_loss_dec', dtype=tf.float32)


@tf.function
def test_step(cover_val, hidden_val):
    encoded_imgs = encoder_model([cover_val, hidden_val], training=True)
    decoded_imgs = decoder_model(encoded_imgs, training=True)

    enc_loss = rgb_diff(encoded_imgs, cover_val)
    dec_loss = rgb_diff(decoded_imgs, hidden_val)

    val_loss_enc(enc_loss)
    val_loss_dec(dec_loss)

def train(cover_ds, hidden_ds, epochs):
  for epoch in range(epochs):
    start = time.time()
    cover_it = iter(cover_ds)
    hidden_it = iter(hidden_ds)
    for cover_batch, hidden_batch in zip(cover_it, hidden_it):
      train_step(cover_batch, hidden_batch)
    
    
    cover_it_val = iter(ds_cover_val)
    hidden_it_val = iter(ds_hidden_val)
    for cover_val_batch, hidden_val_batch in zip(cover_it_val, hidden_it_val):
        test_step(cover_val_batch, hidden_val_batch)
    
    with encoder_summary_writer.as_default():
        tf.summary.scalar('train_loss_enc', train_loss_enc.result(), step=epoch)
        tf.summary.scalar('val_loss_enc', val_loss_enc.result(), step=epoch)
        
    with decoder_summary_writer.as_default():
        tf.summary.scalar('train_loss_dec', train_loss_dec.result(), step=epoch)
        tf.summary.scalar('val_loss_dec', val_loss_dec.result(), step=epoch)
    
    with encoder_summary_writer.as_default():
        # Don't forget to reshape.
        cover_img = ds_cover_val[0];
        hidden_img = ds_hidden_val[0];
        encoded_img = encoder_model([cover_img, hidden_img], training=False)
        decoded_img = decoder_model(encoded_img[0], training=False)
        images = np.reshape([cover_img, hidden_img, encoded_img, decoded_img], (-1, 96, 96, 3))
        tf.summary.image("Images at epoch: " + epoch, images, max_outputs=4, step=0)
    

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
EPOCHS = 5
train(ds_cover_train, ds_hidden_train, EPOCHS)