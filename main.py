import os
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
from data import get_data
from models import get_model
import tensorflow_federated as tff
from absl import app
from absl import flags
from datetime import datetime
import six
import shutil
import os

flags.DEFINE_string("eval_mode", "train", "Which evaluation mode")
flags.DEFINE_integer("gpuid", 0, "Which gpu id to use")

# Training flags
flags.DEFINE_string("net", "lenet_fc", "Which net to use.")
flags.DEFINE_string("mode", "base", "Which mode to use.")
flags.DEFINE_string("cnn_type", "vgg19", "Which mode to use.")
flags.DEFINE_string("data", "drd", "Which dataset to use.")
flags.DEFINE_string("exp_name", None, "Name of experiment")
flags.DEFINE_integer("batch_size", 15, "Batch size")
flags.DEFINE_integer("num_samples", 600, "# of samples (per class) used in training")
flags.DEFINE_integer("num_rounds", 50, "# of rounds in federated learning")
flags.DEFINE_integer("num_examples_per_user", 1000, "No of examples per user")
flags.DEFINE_integer("num_classes", 5, "No of classes")
flags.DEFINE_integer("save_freq", 20, "Saving frequency for model")
flags.DEFINE_integer("n_epochs", 200, "No of epochs")
flags.DEFINE_integer("directory", None, "Train directory")

flags.DEFINE_integer("seed", None, "Random seed.")
flags.DEFINE_boolean("use_fl", True, "Use federated learning or not")

# Attack flags
flags.DEFINE_string("attack", "pgd", "Which attack to use.")
flags.DEFINE_string("attack_source", "base", "Source model for attack.")
flags.DEFINE_string("attack_ord", "inf", "L_inf/ L_2.")
flags.DEFINE_integer("pgd_steps", 40, "No of pgd steps")
flags.DEFINE_float("adv_weight", 2.0, "Weight for adversarial cost")
flags.DEFINE_float("eps", 0.03, "Epsilon for attack")
flags.DEFINE_float("vul_weight", 1.0, "Vulnerability weight")
flags.DEFINE_float("step_size", 0.007, "Step size for attack")
flags.DEFINE_boolean("white_box", True, "White box/black box attack")

FLAGS = app.flags.FLAGS


#model = get_model()
def save_model(state, path, name):
  _model = get_keras_model(state)
  _model.save(path + "/" + name + ".h5")

def get_keras_model(state):
  keras_model = get_model(FLAGS)
  tff.learning.assign_weights_to_keras_model(keras_model, state.model)
  return keras_model

def main(argv):
  tf.compat.v1.enable_v2_behavior()

  # For high-performance executor stack
  if six.PY3:
    tff.framework.set_default_executor(tff.framework.create_local_executor())

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

  train_data, test_data = get_data(FLAGS)
  if FLAGS.use_fl:
    if FLAGS.data in ("drd", "tbc"):
      for x, y in train_data[0].take(1):
        sample_batch = {'x': x.numpy(), 'y': y.numpy()}
    else:
      sample_batch = train_data[0][-1]

    def model_fn():
        model = get_model(FLAGS)
        return tff.learning.from_compiled_keras_model(model, sample_batch)
    fed_avg = tff.learning.build_federated_averaging_process(model_fn)
    state = fed_avg.initialize()

    print("Round starts!")
    start_time = datetime.now()
    for round_num in range(1, FLAGS.num_rounds + 1):
        state, metrics = fed_avg.next(state, train_data)
        print('round {:2d}, metrics={} Elasped time: {}'.format(round_num, metrics, datetime.now()-start_time))
        save_model(state, result_dir, "round_{}".format(round_num))

    keras_model = get_keras_model(state)

    if FLAGS.data == "cifar10":
      (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
      train_images, test_images = train_images / 255.0, test_images / 255.0
      score = keras_model.evaluate(test_images, test_labels, verbose=0)
    elif FLAGS.data in ("drd", "tbc"):
      score = keras_model.evaluate(test_data, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

if __name__ == '__main__':
    app.run(main)

