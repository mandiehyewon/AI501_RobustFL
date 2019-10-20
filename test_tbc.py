import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from get_dataset import *
from glob import glob

# configure
experiment = "all"
dataset_type = "tbc"
cnn_type = "vgg19" # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"
#random_seed = 42
HEIGHT = 224
WIDTH = 224
CLASSES = 2
BATCH_SIZE = 32
num_classes=2
pretrained = "imagenet" 
feature_extraction = False

dataset_path = "/st2/myung/data/TBc"
all_train = "{}/ALL/train/".format(dataset_path)
all_test = "{}/ALL/test/".format(dataset_path)
ms_train = "{}/MS/train/".format(dataset_path)
ms_test = "{}/MS/test/".format(dataset_path)
cs_train = "{}/CS/train/".format(dataset_path)
cs_test = "{}/CS/test/".format(dataset_path)
cs_test_files = "{}/CS/test/**/*.png".format(dataset_path)

#test_X, test_y = get_tbc_Xy(glob(cs_test_files))

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
    all_test,
    target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

model = tf.keras.models.load_model('{}_{}_{}_{}_{}_img{}x{}.h5'.format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, HEIGHT, WIDTH
))

predict_classes = []
test_y = []
eval_X = []
eval_y = []
for i in range(len(cs_test_files)//BATCH_SIZE+2):
  x,y = next(test_generator)
  predict = model.predict(x)
  predict_classes.append(np.argmax(predict, axis=-1))
  test_y.append(np.argmax(y, axis=-1))
  eval_X.append(x)
  eval_y.append(y)


test_y = np.concatenate(test_y, axis=0)
predict_classes = np.concatenate(predict_classes, axis=0)
eval_X = np.concatenate(eval_X, axis=0)
eval_y = np.concatenate(eval_y, axis=0)

print(test_y.shape, predict_classes.shape)
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_y, predict_classes, target_names=target_names))


score = model.evaluate(eval_X,eval_y, verbose=1)
print()
print('val loss:', score[0])
print('val accuracy:', score[1])
