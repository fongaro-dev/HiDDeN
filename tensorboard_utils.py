import os
import io
import typing as t
import numpy as np
from math import ceil
from tqdm import tqdm
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras import layers

class Summary:
  """ Helper class to write TensorBoard summaries """

  def __init__(self, output_dir: str):
    self.dpi = 120
    plt.style.use('seaborn-deep')

    self.writers = [
        tf.summary.create_file_writer(output_dir),
        tf.summary.create_file_writer(os.path.join(output_dir, 'test'))
    ]

  def get_writer(self, training: bool):
    return self.writers[0 if training else 1]

  def scalar(self, tag, value, step: int = 0, training: bool = False):
    writer = self.get_writer(training)
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def image(self, tag, values, step: int = 0, training: bool = False):
    writer = self.get_writer(training)
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=len(values))

  def figure(self,
             tag,
             figure,
             step: int = 0,
             training: bool = False,
             close: bool = True):
    """ Write matplotlib figure to summary
    Args:
      tag: data identifier
      figure: matplotlib figure or a list of figures
      step: global step value to record
      training: training summary or test summary
      close: flag to close figure
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, dpi=self.dpi, format='png', bbox_inches='tight')
    buffer.seek(0)
    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    self.image(tag, tf.expand_dims(image, 0), step=step, training=training)
    if close:
      plt.close(figure)

  def image_cycle(self,
                  tag: str,
                  images: t.List[np.ndarray],
                  labels: t.List[str],
                  step: int = 0,
                  training: bool = False):
    """ Plot image cycle to TensorBoard
    Args:
      tags: data identifier
      images: list of np.ndarray where len(images) == 3 and each array has
              shape (N,H,W,C)
      labels: list of string where len(labels) == 3
      step: global step value to record
      training: training summary or test summary
    """
    assert len(images) == len(labels) == 4
    for sample in range(len(images[0])):
      figure, axes = plt.subplots(nrows=2,
                                  ncols=2,
                                  figsize=(9, 9),
                                  dpi=self.dpi)
      axes[0, 0].imshow(images[0][sample, ...], interpolation='none')
      axes[0, 0].set_title(labels[0])

      axes[0, 1].imshow(images[1][sample, ...], interpolation='none')
      axes[0, 1].set_title(labels[1])

      axes[1, 0].imshow(images[2][sample, ...], interpolation='none')
      axes[1, 0].set_title(labels[2])

      axes[1, 1].imshow(images[3][sample, ...], interpolation='none')
      axes[1, 1].set_title(labels[3])

      plt.setp(axes, xticks=[], yticks=[])
      plt.tight_layout()
      figure.subplots_adjust(wspace=0.02, hspace=0.02)
      self.figure(tag=f'{tag}/sample_#{sample:03d}',
                  figure=figure,
                  step=step,
                  training=training,
                  close=True)


def append_dict(dict1: dict, dict2: dict, replace: bool = False):
  """ append items in dict2 to dict1 """
  for key, value in dict2.items():
    if replace:
      dict1[key] = value
    else:
      if key not in dict1:
        dict1[key] = []
      dict1[key].append(value)