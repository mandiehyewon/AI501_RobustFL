import os
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import *
from get_dataset import *
from art.classifiers import KerasClassifier
from sklearn.metrics import classification_report
from art.attacks import FastGradientMethod, CarliniLInfMethod, ProjectedGradientDescent, CarliniL2Method
# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32
dataset_type = "tbc"
experiment = "all"
cnn_type = "vgg19"
best_acc = 0.864
example = 0
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


    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")])

    print(cnn.summary())
    if not os.path.isfile('model.h5'):
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
        target_names = ["Class {}".format(i) for i in range(CLASSES)]
        print(classification_report(y_test, predict_classes, target_names=target_names))


        score = cnn.evaluate(x_test, y_test,verbose=1)
        print()
        print('test loss:', score[0])
        print('test accuracy:', score[1])

        cnn.save('model.h5'.format(score[1]))
    else:
        cnn.load_weights('model.h5'.format(best_acc))
        print("Loaded model")


    # plot result
    #import pdb
    #pdb.set_trace()
#     print()
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
    #tf.compat.v1.disable_eager_execution()

    count = 0
    for x,y in train_ds.take(1000):
        if not count:
            x_train = np.array(x, dtype=np.float32)
            y_train = np.array(y, dtype=np.float32)
            count += 1
        else:
            x_train = np.append(x_train, x, axis=0)
            y_train = np.append(y_train, y, axis=0)

    robust_cnn = KerasClassifier(model=cnn, clip_values=(0, 1))

    print("------Evaluating standard classifier-----")
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on benign test examples: {:.3f}%'.format(accuracy * 100))
    x_example = x_test[example]

    # Generate adversarial test examples
    # Evaluate the ART classifier on adversarial test examples
    print("*"*100)
    attack = FastGradientMethod(classifier=robust_cnn, eps=0.03)
    x_test_adv = attack.generate(x=x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on FastGradientMethod test examples: {:.3f}%'.format(accuracy * 100))
    fg_example = x_test_adv[example]

    print("*"*100)
    attack = CarliniLInfMethod(classifier=robust_cnn, eps=0.03, max_iter=100, learning_rate=0.01)
    x_test_adv = attack.generate(x=x_test, y=tf.keras.utils.to_categorical(np.ones_like(y_test)))
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on CarliniLInfMethod test examples: {:.3f}%'.format(accuracy * 100))
    carlini_example = x_test_adv[example]

    print("*"*100)
    attack = ProjectedGradientDescent(robust_cnn, norm=np.inf, eps=0.03, eps_step=0.007,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=inf] test examples: {:.3f}%'.format(accuracy * 100))
    pgdInf_example = x_test_adv[example]

    print("*"*100)
    attack = ProjectedGradientDescent(robust_cnn, norm=1, eps=12, eps_step=0.1,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=1] test examples: {:.3f}%'.format(accuracy * 100))
    pgd1_example = x_test_adv[example]

    print("*"*100)
    attack = ProjectedGradientDescent(robust_cnn, norm=2, eps=0.5, eps_step=0.05,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=2] test examples: {}%'.format(accuracy * 100))
    pgd2_example = x_test_adv[example]

    imgs = [x_example, fg_example, carlini_example,\
            pgdInf_example, pgd1_example, pgd2_example]
    img_labels = ["Original", "FastGradient", "CarliniInf",\
                  "PGD[norm=Inf]", "PGD[norm=1]", "PGD[norm=2]"]
    fig=plt.figure(figsize=(20, 20))
    columns = 6
    rows = 1
    for i in range(0, columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        plt.title(img_labels[i])

    plt.savefig("standard.png")


    attack = ProjectedGradientDescent(robust_cnn, norm=np.inf, eps=0.03, eps_step=0.007,max_iter=40)
    x_adv = attack.generate(x=x_train)

    print("------Training Robust Model------")
    x_train_final = np.append(x_train, x_adv, axis=0)
    y_train_final = np.append(y_train, y_train, axis=0)

    if not os.path.isfile('robust_model.h5'):
      print(x_adv.shape, y_train.shape, x_train_final.shape, y_train_final.shape, x_train.shape)
      robust_model = robust_cnn.fit(x_train_final, y_train_final, nb_epochs=1, batch_size=128)
      robust_cnn.save('robust_model.h5')
    else:
        cnn.load_weights('robust_model.h5')
        print("Loaded Robust model")

    print("------Evaluating robust classifier-----")
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on benign test examples: {:.3f}%'.format(accuracy * 100))
    x_example = x_test[example]

    # Generate adversarial test examples
    # Evaluate the ART classifier on adversarial test examples
    print("*"*100)
    attack = FastGradientMethod(classifier=robust_cnn, eps=0.03)
    x_test_adv = attack.generate(x=x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on FastGradientMethod test examples: {:.3f}%'.format(accuracy * 100))
    fg_example = x_test_adv[example]

    print("*"*100)
    attack = CarliniLInfMethod(classifier=robust_cnn, eps=0.03, max_iter=100, learning_rate=0.01)
    x_test_adv = attack.generate(x=x_test, y=tf.keras.utils.to_categorical(np.ones_like(y_test)))
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on CarliniLInfMethod test examples: {:.3f}%'.format(accuracy * 100))
    carlini_example = x_test_adv[example]

    print("*"*100)
    attack = ProjectedGradientDescent(robust_cnn, norm=np.inf, eps=0.03, eps_step=0.007,max_iter=40)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=inf] test examples: {:.3f}%'.format(accuracy * 100))
    pgdInf_example = x_test_adv[example]

    print("*"*100)
    attack = ProjectedGradientDescent(robust_cnn, norm=1, eps=12, eps_step=0.1,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=1] test examples: {:.3f}%'.format(accuracy * 100))
    pgd1_example = x_test_adv[example]

    print("*"*100)
    attack = ProjectedGradientDescent(robust_cnn, norm=2, eps=0.5, eps_step=0.05,max_iter=100)
    x_test_adv = attack.generate(x_test)
    perturbation = np.mean(np.abs((x_test_adv - x_test)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    predict = robust_cnn.predict(x_test_adv)
    predict_classes = np.argmax(predict, axis=-1)
    target_names = ["Class {}".format(i) for i in range(CLASSES)]
    print(classification_report(y_test, predict_classes, target_names=target_names))
    accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    print('Accuracy on ProjectedGradientDescent[norm=2] test examples: {}%'.format(accuracy * 100))
    pgd2_example = x_test_adv[example]

    imgs = [x_example, fg_example, carlini_example,\
            pgdInf_example, pgd1_example, pgd2_example]
    img_labels = ["Original", "FastGradient", "CarliniInf",\
                  "PGD[norm=Inf]", "PGD[norm=1]", "PGD[norm=2]"]
    fig=plt.figure(figsize=(20, 20))
    columns = 6
    rows = 1
    for i in range(0, columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        plt.title(img_labels[i])

    plt.savefig("robust.png")
