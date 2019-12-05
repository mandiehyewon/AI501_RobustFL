import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from models import *
from data import get_data
from models import get_model
import tensorflow_federated as tff
from absl import app
from absl import flags
from datetime import datetime
import six
import shutil
import os
import pdb


# Training
flags.DEFINE_string("cnn_type", "vgg19", "cnn1, cnn3, cnn4, vgg16, vgg19, resnet50,inception_v3 coulde be use.")
flags.DEFINE_string("data", "tbc", "Which dataset to use.")
flags.DEFINE_string("exp_name", "fl_baseline", "Name of experiment")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_samples", 600, "# of samples (per class) used in training")
flags.DEFINE_integer("num_rounds", 5, "# of rounds in federated learning")
flags.DEFINE_integer("num_div", 1, "# of division in china set")
flags.DEFINE_integer("num_examples_per_user", 1000, "No of examples per user")
flags.DEFINE_integer("num_classes", 2, "No of classes")
flags.DEFINE_integer("save_freq", 20, "Saving frequency for model")
flags.DEFINE_integer("n_epochs", 1, "No of epochs")
flags.DEFINE_integer("directory", None, "Train directory")
flags.DEFINE_integer("width", 224, "Width of the image")
flags.DEFINE_integer("height", 224, "Height of the image")
flags.DEFINE_integer("seed", None, "Random seed.")
flags.DEFINE_boolean("use_fl", True, "Use federated learning or not")


# Attack flags
flags.DEFINE_boolean("attack", True, "Attack to use.")
flags.DEFINE_string("attack_source", "base", "Source model for attack.")
flags.DEFINE_string("attack_ord", "inf", "L_inf/ L_2.")
flags.DEFINE_integer("pgd_steps", 40, "No of pgd steps")
flags.DEFINE_float("adv_weight", 2.0, "Weight for adversarial cost")
flags.DEFINE_float("eps", 0.03, "Epsilon for attack")
flags.DEFINE_float("vul_weight", 1.0, "Vulnerability weight")
flags.DEFINE_float("step_size", 0.007, "Step size for attack")
flags.DEFINE_boolean("white_box", True, "White box/black box attack")


FLAGS = app.flags.FLAGS


def main(argv):

    np.random.seed(0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Backup current sources to experiment source folder
    exp_dir = os.path.join("experiments", FLAGS.exp_name + "_" + str(datetime.now()))
    result_dir = os.path.join(exp_dir, "result")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copytree(
        os.path.abspath(os.path.curdir),
        os.path.join(exp_dir, "source"),
        ignore=lambda src, names: {
        "datasets",
        ".vscode",
        "__pycache__",
        ".git",
        "*.png",
        "env",
        "experiments",
        "baseline_results"
        }
    )


    if six.PY3:
        tff.framework.set_default_executor(tff.framework.create_local_executor())

    train_data, test_data = get_data(FLAGS)


    def save_model(state, path, name, keras_model):
        tff.learning.assign_weights_to_keras_model(keras_model, state.model)
        keras_model.save(path + "/" + name + ".h5")

    STEPS_PER_EPOCH = 200
    for x, y in train_data[0].take(1):
        sample_batch = OrderedDict([("x", x.numpy()), ("y",y.numpy())])

    def model_fn():
        cnn = get_model(FLAGS)
        return tff.learning.from_compiled_keras_model(cnn, sample_batch)

    fed_avg = tff.learning.build_federated_averaging_process(model_fn)
    state = fed_avg.initialize()
    keras_model = get_model(FLAGS)
    print(keras_model.summary())
    print("non-trainable: ",len(keras_model.non_trainable_weights))
    print("trainable: ",len(keras_model.trainable_weights))
    state = tff.learning.state_with_new_model_weights(
        state,
        trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
        non_trainable_weights=[
            v.numpy() for v in keras_model.non_trainable_weights
    ])
    print("Round starts!")
    print(len(train_data))
    start_time = datetime.now()
    for round_num in range(1, FLAGS.num_rounds + 1):
        state, metrics = fed_avg.next(state, train_data)
        print('round {:2d}, metrics={} Elasped time: {}'.format(round_num, metrics, datetime.now()-start_time))
        save_model(state, result_dir, "round_{}".format(round_num), keras_model)
        tff.learning.assign_weights_to_keras_model(keras_model, state.model)
        score = keras_model.evaluate(test_data, steps=1, verbose=0)
        print("[evaluation] loss: {}\t accuracy: {}".format(score[0], score[1]))


    tff.learning.assign_weights_to_keras_model(keras_model, state.model)

    score = keras_model.evaluate(test_data,  steps=1, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # evaluate
    print()
    predict_classes = []
    test_y = []
    eval_X = []
    eval_y = []

    for x,y in test_data.take(1):
        predict = keras_model.predict(x)
        predict_classes.append(np.argmax(predict, axis=-1))
        test_y.append(y)
        eval_X.append(x)
        eval_y.append(y)

    test_y = np.concatenate(test_y, axis=0)
    predict_classes = np.concatenate(predict_classes, axis=0)
    eval_X = np.concatenate(eval_X, axis=0)
    eval_y = np.concatenate(eval_y, axis=0)

    print(test_y.shape, predict_classes.shape)
    from sklearn.metrics import classification_report
    target_names = ["Class {}".format(i) for i in range(FLAGS.num_classes)]
    print(classification_report(test_y, predict_classes, target_names=target_names))


if __name__ == "__main__":
    app.run(main)
