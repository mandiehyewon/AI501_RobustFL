import os
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
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

flags.DEFINE_string("eval_mode", "train", "Which evaluation mode")
flags.DEFINE_integer("gpuid", 0, "Which gpu id to use")

# Training flags
flags.DEFINE_string("net", "lenet_fc", "Which net to use.")
flags.DEFINE_string("mode", "base", "Which mode to use.")
flags.DEFINE_string("cnn_type", "cnn3", "Which mode to use.")
flags.DEFINE_string("data", "cifar10", "Which dataset to use.")
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

def main(argv):
  tf.compat.v1.enable_v2_behavior()

  # For high-performance executor stack
  #if six.PY3:
   # tff.framework.set_default_executor(tff.framework.create_local_executor())

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

    # TODO: fix evaluate
    def evaluate(state, round_num):
        #tff.learning.assign_weights_to_keras_model(keras_model, state.model) # for tff 0.9.0
        model = get_model(FLAGS)
        model.set_weights(tff.learning.keras_weights_from_tff_weights(state.model))
        model.evaluate(test_data, steps=2)

    fed_avg = tff.learning.build_federated_averaging_process(model_fn)
    state = fed_avg.initialize()
    print("Round starts!")
    start_time = datetime.now()
    for round_num in range(1, FLAGS.num_rounds + 1):
        state, metrics = fed_avg.next(state, train_data)
        print('round {:2d}, metrics={} Elasped time: {}'.format(round_num, metrics, datetime.now()-start_time))

    #evaluate(state, FLAGS.num_rounds + 1)
    #evaluation = tff.learning.build_federated_evaluation(fed_avg)
    #train_metrics = evaluation(state.model, train_data)
    #test_metrics = evaluation(state.model, test_data)
    #print("Training metrics", train_metrics)
    #print("Testing metrics", test_metrics)
  
  else:
    model = get_model(FLAGS)
    result = model.fit(
        train_data[0],
        train_data[1],
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.n_epochs,
        verbose=1,
    )
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save("{}_{}.h5".format(
            FLAGS.cnn_type, FLAGS.data))

if __name__ == '__main__':
    app.run(main)

