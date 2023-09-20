# %%
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import _keras

from tensorboard_utils import *

from dataset_utils import split_dataset

from tqdm import tqdm

import numpy as np
import os
import os, stat
import shutil
import time

from models import *
from losses import rgb_diff

# %% [markdown]
# ## Load the dataset

# %%
def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

print("Loading dataset")
split0, split1 = tfds.even_splits('all', n=2)
ds_cover = tfds.load('stl10', split=split0 , shuffle_files=True)
ds_hidden = tfds.load('stl10', split=split1 , shuffle_files=True)
print("Loaded dataset")


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH_SIZE = 32
strategy = tf.distribute.MirroredStrategy(devices=None)
GLOBAL_BATCH_SIZE = strategy.num_replicas_in_sync * BATCH_SIZE

print(f'Number of devices: {strategy.num_replicas_in_sync}')

[ds_cover_train,  ds_cover_val , ds_cover_test] = split_dataset(ds_cover)
[ds_hidden_train, ds_hidden_val, ds_hidden_test] = split_dataset(ds_hidden)


print(f"Length of clear train set: {len(ds_cover_train)}, validation: {len(ds_cover_val)}, and test: {len(ds_cover_test)}")
print(f"Length of hidden train set: {len(ds_hidden_train)}, validation: {len(ds_hidden_val)}, and test: {len(ds_hidden_test)}")


NUM_TRAIN_SAMPLES = len(ds_cover_train)
NUM_VAL_SAMPLES = len(ds_hidden_val)
TRAIN_STEPS = ceil(NUM_TRAIN_SAMPLES / GLOBAL_BATCH_SIZE)
TEST_STEPS = ceil(NUM_VAL_SAMPLES / GLOBAL_BATCH_SIZE)

def normalize(ds):
    # print(image)
    image_ds = (tf.cast(ds['image'],tf.float32)) / 255.0
    return image_ds

ds_cover_train = ds_cover_train.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
ds_cover_train = ds_cover_train.cache()
ds_hidden_train = ds_hidden_train.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
ds_hidden_train = ds_hidden_train.cache()

ds_train = tf.data.Dataset.zip((ds_cover_train, ds_hidden_train)).batch(GLOBAL_BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

ds_hidden_val = ds_hidden_val.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
ds_hidden_val = ds_hidden_val.cache()
ds_cover_val = ds_cover_val.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
ds_cover_val = ds_cover_val.cache()

ds_val = tf.data.Dataset.zip((ds_cover_val, ds_hidden_val)).batch(GLOBAL_BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

ds_hidden_test = ds_hidden_test.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
ds_hidden_test = ds_hidden_test.cache()
ds_cover_test = ds_cover_test.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
ds_cover_test = ds_cover_test.cache()

ds_test = tf.data.Dataset.zip((ds_cover_test, ds_hidden_test)).batch(GLOBAL_BATCH_SIZE, ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

plot_ds = tf.data.Dataset.zip(
    (ds_cover_test.take(5).batch(1), ds_hidden_test.take(5).batch(1)))

# create distributed datasets
ds_train = strategy.experimental_distribute_dataset(ds_train)
ds_val = strategy.experimental_distribute_dataset(ds_val)
ds_test = strategy.experimental_distribute_dataset(ds_test)

# %% [markdown]
# ## Define the training steps

# %%

# Create the model within strategy scope
input_shape = (96, 96, 3)  # Adjust input size as needed
with strategy.scope():
    enc_dec_model = create_encoder_decoder_model(input_shape)

    enc_dec_model.summary()

    enc_dec_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                            beta_1=0.5,
                                            beta_2=0.9)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder_optimizer=enc_dec_optimizer,
                                    decoder_encoder=enc_dec_model)



def reduce_mean(per_sample_loss):
    """ return the global mean of per-sample loss """
    return tf.reduce_sum(per_sample_loss) / GLOBAL_BATCH_SIZE

enc_weigh = 0.4
dec_weigh = 0.6
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(cover_imgs, hidden_imgs):
    result = {}
    with tf.GradientTape() as tape:
        [encoded_imgs, decoded_imgs] = enc_dec_model([cover_imgs, hidden_imgs], training=True)

        enc_loss = reduce_mean(rgb_diff(encoded_imgs, cover_imgs))
        dec_loss = reduce_mean(rgb_diff(decoded_imgs, hidden_imgs))

        total_loss = enc_weigh * enc_loss + dec_weigh * dec_loss


    result.update({
    'train/loss_enc': enc_loss,
    'train/loss_dec': dec_loss,
    'train/total_loss': total_loss,
    })

    enc_dec_optimizer.minimize(loss=total_loss, var_list=enc_dec_model.trainable_variables, tape = tape)

    return result


def reduce_dict(d: dict):
  """ inplace reduction of items in dictionary d """
  return {
        k: strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
        for k, v in d.items()
  }

@tf.function
def distributed_train_step(x, y):
    results = strategy.run(train_step, args=(x, y))
    results = reduce_dict(results)
    return results

def test_step(cover_val, hidden_val):
    result = {}
    [encoded_imgs, decoded_imgs] = enc_dec_model([cover_val, hidden_val], training=False)

    enc_loss = reduce_mean(rgb_diff(encoded_imgs, cover_val))
    dec_loss = reduce_mean(rgb_diff(decoded_imgs, hidden_val))

    total_loss = enc_weigh * enc_loss + dec_weigh * dec_loss

    result.update({
    'val/loss_enc': enc_loss,
    'val/loss_dec': dec_loss,
    'val/total_loss': total_loss,
    })

    return result

@tf.function
def distributed_test_step(x, y):
    results = strategy.run(test_step, args=(x, y))
    results = reduce_dict(results)
    return results

def train(ds, summary, epoch: int):
    results = {}
    for cover_batch, hidden_batch in tqdm(ds, total=TRAIN_STEPS):
        result = distributed_train_step(cover_batch, hidden_batch)
        append_dict(results, result)
    for key, value in results.items():
        results[key] = tf.reduce_mean(value)
        summary.scalar(key, results[key], step=epoch, training=True)
    return results


def test(ds, summary, epoch: int):
    results = {}
    for cover_batch, hidden_batch in tqdm(ds, total=TEST_STEPS):
        result = distributed_test_step(cover_batch, hidden_batch)
        append_dict(results, result)
    for key, value in results.items():
        results[key] = tf.reduce_mean(value)
        summary.scalar(key, results[key], step=epoch, training=False)
    return results

# %% [markdown]
# ## Debugging functions

# %%

def plot_cycle(ds, summary, epoch: int):
    """ plot X -> G(X) -> F(G(X)) and Y -> F(Y) -> G(F(Y)) """
    samples = {}
    for cover_img, hidden_img in ds:
        [encoded_img, decoded_img] = enc_dec_model([cover_img, hidden_img], training=False)
        append_dict(dict1=samples,
                    dict2={
                        'cover_img': cover_img,
                        'hidden_img': hidden_img,
                        'encoded_img': encoded_img,
                        'decoded_img': decoded_img
                    })
    for key, images in samples.items():
        # scale images back to [0, 255]
        images = tf.concat(images, axis=0).numpy()
        images = (images * 255.0).astype(np.uint8)
        samples[key] = images
    summary.image_cycle(
        tag=f'Encoding Cycle',
        images=[samples['cover_img'], samples['hidden_img'], samples['encoded_img'], samples['decoded_img']],
        labels=['cover_img', 'hidden_img', 'encoded_img', 'decoded_img'],
        step=epoch,
        training=False)


OUTPUT_DIR = 'runs'  # directory to store checkpoint and TensorBoard summary

if os.path.exists(OUTPUT_DIR):
    # Clear out any prior log data.
    shutil.rmtree(OUTPUT_DIR, onerror=remove_readonly)
os.makedirs(OUTPUT_DIR)

summary = Summary(output_dir=OUTPUT_DIR)

# %% [markdown]
# # Train

# %% [markdown]
# ### (Optional) Load Checkpoint

# %%
CKPT_PATH = "./ckpt"

# %%
NUM_EPOCHS = 20
checkpoint_manager = tf.train.CheckpointManager(checkpoint, CKPT_PATH, max_to_keep=3)

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1:03d}/{NUM_EPOCHS:03d}')
    start = time.time()
    train_results = train(ds_train, summary, epoch)
    test_results = test(ds_val, summary, epoch)
    end = time.time()

    print(f'train/loss_enc: {train_results["train/loss_enc"]:.04f}\t\t'
            f'train/loss_dec: {train_results["train/loss_dec"]:.04f}\n'
            f'train/total_loss: {train_results["train/total_loss"]:.04f}\n'
            f'val/loss_enc: {test_results["val/loss_enc"]:.04f}\t\t'
            f'val/loss_dec: {test_results["val/loss_dec"]:.04f}\n'
            f'val/total_loss: {test_results["val/total_loss"]:.04f}\n'
            f'Elapse: {end - start:.02f}s\n')

    if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
        save_path = checkpoint_manager.save()
        plot_cycle(plot_ds, summary, epoch)


