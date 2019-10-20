import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from models import *
from get_dataset import *
from data import get_data
from models import get_model
import tensorflow_federated as tff
from absl import app
from absl import flags
from datetime import datetime
import six
import shutil
import os
from art.classifiers import TensorFlowV2Classifier
from art.attacks import FastGradientMethod, CarliniLInfMethod

flags.DEFINE_string("eval_mode", "train", "Which evaluation mode")
flags.DEFINE_integer("gpuid", 0, "Which gpu id to use")

# Training flags
flags.DEFINE_string("net", "lenet_fc", "Which net to use.")
flags.DEFINE_string("mode", "base", "Which mode to use.")
flags.DEFINE_string("cnn_type", "vgg19", "Which mode to use.")
flags.DEFINE_string("data", "tbc", "Which dataset to use.")
flags.DEFINE_string("exp_name", None, "Name of experiment")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_samples", 600, "# of samples (per class) used in training")
flags.DEFINE_integer("num_rounds", 5, "# of rounds in federated learning")
flags.DEFINE_integer("num_div", 1, "# of division in china set")
flags.DEFINE_integer("num_examples_per_user", 1000, "No of examples per user")
flags.DEFINE_integer("num_classes", 5, "No of classes")
flags.DEFINE_integer("save_freq", 20, "Saving frequency for model")
flags.DEFINE_integer("n_epochs", 1, "No of epochs")
flags.DEFINE_integer("directory", None, "Train directory")

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

# configure
experiment = "all"
dataset_type = "tbc"
cnn_type = "vgg19" # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"
#random_seed = 42
total_epoch=5
batch_size=32


def main(argv):
    tf.compat.v1.enable_v2_behavior()

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
        }
    )
    if six.PY3:
        tff.framework.set_default_executor(tff.framework.create_local_executor())
    from glob import glob
    dataset_path = "/st2/myung/data/TBc"
    all_train = "{}/ALL/train/".format(dataset_path)
    all_test = "{}/ALL/test/".format(dataset_path)
    ms_train = "{}/MS/train/".format(dataset_path)
    ms_test = "{}/MS/test/".format(dataset_path)
    cs_train = "{}/CS/train/".format(dataset_path)
    cs_test = "{}/CS/test/".format(dataset_path)

    train_data, test_data = get_data(FLAGS)
    HEIGHT = 224
    WIDTH = 224
    CLASSES = 1
    BATCH_SIZE = 32

    def get_model():
      base_model = tf.keras.applications.VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))

      x = base_model.output
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(512, activation='relu')(x)
      x = tf.keras.layers.Dropout(0.4)(x)
      predictions = tf.keras.layers.Dense(CLASSES, activation='sigmoid')(x)
      cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # transfer learning
      for layer in base_model.layers:
        layer.trainable = False

      cnn.compile(
        loss=tf.keras.losses.BinaryCrossentropy("loss"),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=[tf.keras.metrics.BinaryAccuracy("acc")],)
      return cnn

    def get_keras_model(state):
      keras_model = get_model()
      tff.learning.assign_weights_to_keras_model(keras_model, state.model)
      return keras_model

    def save_model(state, path, name):
      _model = get_keras_model(state)
      _model.save(path + "/" + name + ".h5")

    EPOCHS = 5
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 320

    for x, y in train_data[0].take(1):
      sample_batch = {'x': x.numpy(), 'y': y.numpy()}

    def model_fn():
      cnn = get_model()
      return tff.learning.from_compiled_keras_model(cnn, sample_batch)

    fed_avg = tff.learning.build_federated_averaging_process(model_fn)
    state = fed_avg.initialize()

    print("Round starts!")
    print(len(train_data))

    start_time = datetime.now()
    for round_num in range(1, FLAGS.num_rounds + 1):
      state, metrics = fed_avg.next(state, train_data)
      print('round {:2d}, metrics={} Elasped time: {}'.format(round_num, metrics, datetime.now()-start_time))
      save_model(state, result_dir, "round_{}".format(round_num))

    keras_model = get_keras_model(state)
    score = keras_model.evaluate(test_data, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    """
    accuracy = result.history['acc']
    test_accuracy = result.history['val_acc']
    loss = result.history['loss']
    test_loss = result.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'red', label='Training accuracy')
    plt.plot(epochs, test_accuracy, 'blue', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_{}_img{}x{}_accuracy.pdf".format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    plt.close()

    plt.plot(epochs, test_accuracy, 'blue', label='Validation accuracy')
    plt.title('Validation accuracy')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_{}_img{}x{}_test_accuracy.pdf".format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    plt.close()

    plt.plot(epochs, accuracy, 'red', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_{}_img{}x{}_train_accuracy.pdf".format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    plt.close()


    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, test_loss, 'blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_{}_img{}x{}_loss.pdf".format(
        experiment, pretrained, cnn_type, feature_extraction, dataset_type, img_rows, img_cols
    ))
    plt.close()

    plt.plot(epochs, test_loss, 'blue', label='Validation loss')
    plt.title('Validation loss')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_{}_img{}x{}_test_loss.pdf".format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    plt.close()

    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_{}_img{}x{}_train_loss.pdf".format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    plt.close()

    # classification report
    predicted_classes = cnn.predict_classes(test_X)

    correct = (predicted_classes == test_y).nonzero()[0]
    incorrect = (predicted_classes != test_y).nonzero()[0]

    from sklearn.metrics import classification_report

    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(test_y, predicted_classes, target_names=target_names))

    # plot correctly predicted classes
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(
            test_X[correct].reshape(img_rows, img_cols, img_channels),
            cmap="gray",
            interpolation="none",
        )
        plt.title(
            "Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct])
        )
        plt.tight_layout()
    plt.savefig(
        "{}_{}_{}_{}_{}_img{}x{}_correct_prediction.pdf".format(
            experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
        )
    )
    plt.close()

    # plot incorrectly predicted classes
    for i, incorrect in enumerate(incorrect[0:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(
            test_X[incorrect].reshape(img_rows, img_cols, img_channels),
            cmap="gray",
            interpolation="none",
        )
        plt.title(
            "Predicted {}, Class {}".format(
                predicted_classes[incorrect], test_y[incorrect]
            )
        )
        plt.tight_layout()
    plt.savefig(
        "{}_{}_{}_{}_{}_img{}x{}_incorrect_prediction.pdf".format(
            experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
        )
    )
    plt.close()
    """

if __name__ == "__main__":
   app.run(main)
