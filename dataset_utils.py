import tensorflow as tf
import tensorflow_datasets as tfds



def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, shuffle=True):
    # Calculate the total number of samples in the dataset
    total_samples = len(dataset)

    # Shuffle the dataset if required
    if shuffle:
        dataset = dataset.shuffle(buffer_size=total_samples, seed=42)

    # Calculate the sizes of each split
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = int(total_samples * test_ratio)

    # Split the dataset
    train_dataset = dataset.take(train_size)
    remaining_dataset = dataset.skip(train_size)
    val_dataset = remaining_dataset.take(val_size)
    test_dataset = remaining_dataset.skip(val_size)

    # Apply batching and prefetching
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset