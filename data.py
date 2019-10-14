import tensorflow as tf
import numpy as np


def get_data_for_fl(FLAGS, source, digit):
  output_sequence = []
  all_samples = [i for i, d in enumerate(source[1]) if d == digit]
  for i in range(0, min(len(all_samples), FLAGS.num_examples_per_user), FLAGS.batch_size):
    batch_samples = all_samples[i:i + FLAGS.batch_size]
    output_sequence.append({
        'x': np.array([source[0][i] / 255.0 for i in batch_samples],
                      dtype=np.float32),
        'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
  return output_sequence


def get_data(FLAGS):
  if FLAGS.data == "mnist":
    train, test = tf.keras.datasets.mnist.load_data()
    FLAGS.num_classes = 10
  if FLAGS.data == "fmnist":
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    FLAGS.num_classes = 10
  if FLAGS.data == "cifar10":
    train, test = tf.keras.datasets.cifar10.load_data()
    FLAGS.num_classes = 10
  
  if FLAGS.use_fl:
    train = [get_data_for_fl(FLAGS, train, d) for d in range(FLAGS.num_classes)]
    test = [get_data_for_fl(FLAGS, test, d) for d in range(FLAGS.num_classes)]
  else:
    train /= 255
    test /= 255

  return train, test


