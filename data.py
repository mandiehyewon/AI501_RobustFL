import tensorflow as tf
import numpy as np
from collections import OrderedDict

from get_dataset import get_dataset_drd, get_dataset_tbc

def get_data_for_fl(FLAGS, source, digit):
  output_sequence = []
  all_samples = [i for i, d in enumerate(source[1]) if d == digit]
  for i in range(0, min(len(all_samples), FLAGS.num_examples_per_user), FLAGS.batch_size):
    batch_samples = all_samples[i:i + FLAGS.batch_size]
    output_sequence.append(OrderedDict([('x', np.array([source[0][i] / 255.0 for i in batch_samples], dtype=np.float32)),
                                        ('y', np.array([source[1][i] for i in batch_samples], dtype=np.int32))]))
  return output_sequence


def get_data(FLAGS):
  if FLAGS.data == "mnist":
    train, test = tf.keras.datasets.mnist.load_data()
    FLAGS.num_classes = 10
  elif FLAGS.data == "fmnist":
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    FLAGS.num_classes = 10
  elif FLAGS.data == "cifar10":
    train, test = tf.keras.datasets.cifar10.load_data()
    FLAGS.num_classes = 10
  elif FLAGS.data == "drd":
    assert FLAGS.num_samples >= FLAGS.batch_size
    train = [get_dataset_drd(d, FLAGS.n_epochs, FLAGS.batch_size, FLAGS.num_samples, split="train") for d in range(FLAGS.num_classes)]
    # test = [get_dataset_drd(d, FLAGS.n_epochs, FLAGS.batch_size, FLAGS.num_samples, split="val") for d in range(FLAGS.num_classes)]
    test = get_dataset_drd(None, 1, FLAGS.batch_size, 5*FLAGS.num_samples, split="val")
  elif FLAGS.data == "tbc":
    assert FLAGS.num_samples >= FLAGS.batch_size
    train_m, test_m = get_dataset_tbc(FLAGS.n_epochs, 1, FLAGS.batch_size, FLAGS.num_samples, center="MS")
    train_c, test_c = get_dataset_tbc(FLAGS.n_epochs, 1, FLAGS.batch_size, FLAGS.num_samples, num_division=FLAGS.num_div, center="CS")
    train = train_m + train_c
    print(len(train_m), len(train_c))
    # test = [test_m, test_c]
    test = test_m.concatenate(test_c)

  if FLAGS.data not in ("drd", "tbc"):
    if FLAGS.use_fl:
      train = [get_data_for_fl(FLAGS, train, d) for d in range(FLAGS.num_classes)]
      test = [get_data_for_fl(FLAGS, test, d) for d in range(FLAGS.num_classes)]
    else:
      train /= 255
      test /= 255

  return train, test
