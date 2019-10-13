# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

if six.PY3:
  tff.framework.set_default_executor(tff.framework.create_local_executor())

#tff.federated_computation(lambda: 'Hello, World!')()

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100

def get_data_for_digit(source, digit):
  output_sequence = []
  all_samples = [i for i, d in enumerate(source[1]) if d == digit]
  for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
    batch_samples = all_samples[i:i + BATCH_SIZE]
    output_sequence.append({
        'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                      dtype=np.float32),
        'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
  return output_sequence

federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

#@test {"output": "ignore"}
from matplotlib import pyplot as plt

plt.imshow(federated_train_data[5][-1]['x'][-1].reshape(28, 28), cmap='gray')
plt.grid(False)
plt.savefig("dataset_example.png")

"""
Creating a model with Keras
"""

def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

"""One critical note on `compile`. When used in the Federated Averaging algorithm,
as below, the `optimizer` is only half of of the total optimization algorithm,
as it is only used to compute local model updates on each client. The rest of
the algorithm involves how these updates are averaged over clients, and how they
are then applied to the global model at the server. In particular, this means
that the choice of optimizer and learning rate used here may need to be
different than the ones you have used to train the model on a standard i.i.d.
dataset. We recommend starting with regular SGD, possibly with a smaller
learning rate than usual. The learning rate we use here has not been carefully
tuned, feel free to experiment.

In order to use any model with TFF, if it is a compiled Keras model,
it can be wrapped with TFF : `tff.learning.from_compiled_keras_model`
"""

def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


"""
Training the model on federated data
"""

#@test {"output": "ignore"}
sample_batch = federated_train_data[5][-1]
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

"""
Invoke the `initialize` computation to construct the server state.
"""

state = iterative_process.initialize()

"""
Execute federated computations

`next()`
- a single round of Federated Averaging
1. consists of pushing the server state (including the model parameters) to the clients
2. on-device training on their local data
3. collecting and averaging model updates
4. producing a new updated model at the server.
- a declarative functional representation of the entire decentralized computation
- some of the inputs are provided by the server (`SERVER_STATE`),
but each participating device contributes its own local dataset.

Let's run a single round of training and visualize the results. We can use the
federated data we've already generated above for a sample of users.
"""


"""
For each round in order to simulate a realistic deployment,
we need to pick a subset of your simulation data from a new randomly selected sample of users

In this example, the same users are reused in each round so the system converges quickly.
"""

#@test {"skip": true}
for round_num in range(1, 10):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))