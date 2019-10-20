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
# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32



# Each image's dimension is 28 x 28

# configure
experiment = "all"
dataset_type = "tbc"
cnn_type = "vgg19" # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"
#random_seed = 42
total_epoch=5
batch_size=32



if __name__ == "__main__":
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
    CLASSES = 2
    BATCH_SIZE = 32

    def get_model():
      base_model = tf.keras.applications.VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))

      x = base_model.output
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(512, activation='relu')(x)
      x = tf.keras.layers.Dropout(0.4)(x)
      predictions = tf.keras.layers.Dense(CLASSES, activation='softmax')(x)
      cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
   
    # transfer learning
      for layer in base_model.layers:
        layer.trainable = False
      
      cnn.compile(
        loss=tf.keras.losses.BinaryCrossentropy("loss"),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=[tf.keras.metrics.Accuracy("acc")],
     )
     return cnn

    def get_keras_model(state):
      keras_model = get_model()
      tff.learning.assign_weights_to_keras_model(keras_model, state.model)
      return keras_model

    EPOCHS = 5
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 320

    cnn = get_model()
    
    for x, y in train_data[0].take(1):
      sample_batch = {'x': x.numpy(), 'y': y.numpy()}

    def model_fn():
      return tff.learning.from_compiled_keras_model(cnn, sample_batch)

    fed_avg = tff.learning.build_federated_averaging_process(model_fn)
    state = fed_avg.initialize()

    print("Round starts!")
    start_time = datetime.now()
    for round_num in range(1, FLAGS.num_rounds + 1):
      state, metrics = fed_avg.next(state, train_data)
      print('round {:2d}, metrics={} Elasped time: {}'.format(round_num, metrics, datetime.now()-start_time))
      save_model(state, result_dir, "round_{}".format(round_num))

    keras_model = get_keras_model(state)
    score = keras_model.evaluate(test_data, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    print()

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
