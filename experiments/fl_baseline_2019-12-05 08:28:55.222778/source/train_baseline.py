import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
import os
from models import *
from get_dataset import *
# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32



# Each image's dimension is 28 x 28

# configure
PATH = "/st2/myung/data/diabetic-retinopathy-detection/kaggle"
dataset_type = "diabetic-retinopathy-detection"
cnn_type = "cnn4" # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"
pretrained = None # "imagenet" or None
transfer_learning = True # True or False (False: pretrained weight is fixed)
random_seed = 42

total_epoch=100
batch_size=128



if __name__ == "__main__":

    tf.compat.v1.set_random_seed(random_seed)

    # Load training and val data into dataframes
    train_ds_all = get_dataset(
        PATH, img_rows, target_class=None, total_epoch=total_epoch, batch_size=batch_size
    )

    val_ds_all = get_dataset(
        PATH, img_rows, target_class=None, total_epoch=total_epoch, batch_size=batch_size, split="val"
    )


    # X forms the training images, and y forms the training labels
    #X = np.array(train_ds_all.iloc[:, 1:])
    #y = np.array(train_ds_all.iloc[:, 0])
    #print(y)
    # X_test forms the test images, and y_test forms the test labels
    #X_test = np.array(val_ds_all.iloc[:, 1:])
    #y_test = np.array(val_ds_all.iloc[:, 0])

    # Display their new shapes
    #print(X.shape, X_val.shape)

    if cnn_type == "cnn1":
        cnn = create_compiled_keras_cnn1_model()
    elif cnn_type == "cnn3":
        cnn = create_compiled_keras_cnn3_model()
    elif cnn_type == "cnn4":
        cnn = create_compiled_keras_cnn4_model()
    elif cnn_type == "vgg16":
        cnn = create_compiled_keras_vgg16_model()
    elif cnn_type == "vgg19":
        cnn = create_compiled_keras_vgg19_model()
    elif cnn_type == "resnet50":
        cnn = create_compiled_keras_resnet50_model()
    elif cnn_type == "inception_v3":
        cnn = create_compiled_keras_inception_v3_model()

    print(cnn.summary())

    # training the model
    result = cnn.fit(train_ds_all,
                    verbose=1,
                    validation_data=val_ds_all)

    score = cnn.evaluate(val_ds_all, verbose=0)
    print('val loss:', score[0])
    print('val accuracy:', score[1])
    cnn.save('{}_{}_{}_{}_img{}x{}.h5'.format(
        cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
    ))

    # plot result

    accuracy = result.history['acc']
    val_accuracy = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'red', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'blue', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_img{}x{}_accuracy.pdf".format(
        cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
    ))
    plt.close()


    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("{}_{}_{}_{}_img{}x{}_loss.pdf".format(
        cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
    ))
    plt.close()
