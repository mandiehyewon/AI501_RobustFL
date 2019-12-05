import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from models import *
from get_dataset_jm import *
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, CarliniLInfMethod, ProjectedGradientDescent
# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32
dataset_type = "tbc"
experiment = "all"
cnn_type ="vgg19"
# configure
HEIGHT = 224
WIDTH = 224
CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 5
STEPS_PER_EPOCH = 200


if __name__ == "__main__":
    tf.compat.v1.enable_v2_behavior()

    from glob import glob
    dataset_path = "/st2/myung/data/TBc"
    all_train = "{}/MSCS/train".format(dataset_path)
    all_test = "{}/MSCS/test".format(dataset_path)
    ms_train = "{}/MS/train".format(dataset_path)
    ms_test = "{}/MS/test".format(dataset_path)
    cs_train = "{}/CS/train".format(dataset_path)
    cs_test = "{}/CS/test".format(dataset_path)

    if "all" in experiment:
        train = all_train
        test = all_test
    elif "ms" in experiment:
        train = ms_train
        test = all_test
    elif "cs" in experiment:
        train = cs_train
        test = all_test

    train_files=glob("{}/**/*.png".format(train))
    test_files=glob("{}/**/*.png".format(test))

    print(train, len(train_files))
    print(test, len(test_files))

    train_ds = get_dataset_tbc_for_single(
        EPOCHS,
        BATCH_SIZE,
        train_files,
        dataset_type="train"
    )

    test_ds = get_dataset_tbc_for_single(
        EPOCHS,
        len(test_files),
        test_files,
        dataset_type="test"
    )

    base_model = tf.keras.applications.VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(CLASSES, activation='softmax')(x)
    cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    """
    cnn.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    """

    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")])

    print(cnn.summary())

    result = cnn.fit_generator(
        train_ds,
        verbose=1,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_ds,
        validation_steps=1)

    # evaluate
    print()
    predict_classes = []
    x_test = []
    y_test = []

    # one batch for validation
    for x,y in test_ds.take(1):
        predict = cnn.predict(x)
        predict_classes = np.argmax(predict, axis=-1)
        x_test = np.array(x, dtype=np.float32)
        y_test = np.array(y, dtype=np.int32)


    print(y_test.shape, predict_classes.shape)
    from sklearn.metrics import classification_report
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))


    score = cnn.evaluate(x_test, y_test,verbose=1)
    print()
    print('test loss:', score[0])
    print('test accuracy:', score[1])

    cnn.save('no_attack_{}_{}_{}_img{}x{}-{:0.3f}.h5'.format(
        experiment, cnn_type, dataset_type, HEIGHT, WIDTH, score[1]
    ))
    # plot result
    #import pdb
    #pdb.set_trace()

    #tf.compat.v1.disable_eager_execution()
    classifier = KerasClassifier(model=cnn, clip_values=(0, 1))

    # Step 5: Evaluate the ART classifier on benign test examples

    predict = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on benign test examples: {:.3f}%'.format(accuracy * 100))

    # Generate adversarial test examples
    # Evaluate the ART classifier on adversarial test examples
    print("*"*100)
    attack = FastGradientMethod(classifier=classifier, eps=0.3)
    x_test_adv = attack.generate(x=x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = classifier.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on FastGradientMethod test examples: {:.3f}%'.format(accuracy * 100))

    print("*"*100)
    attack = CarliniLInfMethod(classifier=classifier, eps=0.3, max_iter=100, learning_rate=0.01)
    x_test_adv = attack.generate(x=x_test, y=tf.keras.utils.to_categorical(np.ones_like(y_test)))
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = classifier.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on CarliniLInfMethod test examples: {:.3f}%'.format(accuracy * 100))

    print("*"*100)
    attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.3, eps_step=0.1,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = classifier.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=inf] test examples: {:.3f}%'.format(accuracy * 100))

    print("*"*100)
    attack = ProjectedGradientDescent(classifier, norm=1, eps=0.3, eps_step=0.1,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predictions = classifier.predict(x_test_adv)
    predict_classes = np.argmax(predictions, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))

    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=1] test examples: {:.3f}%'.format(accuracy * 100))

    print("*"*100)
    attack = ProjectedGradientDescent(classifier, norm=2, eps=0.3, eps_step=0.1,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = classifier.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=2] test examples: {}%'.format(accuracy * 100))
