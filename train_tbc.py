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

    train = all_train
    test = all_test
    #train_X, train_y = get_tbc_Xy(train, file_type="train")
    #val_X, val_y = get_tbc_Xy(test, file_type="test")

    print(train)
    print(test)

    #test_X, test_y = get_tbc_Xy(test, file_type="test")
    HEIGHT = 224
    WIDTH = 224
    CLASSES = 2
    BATCH_SIZE = 32

    # data prep
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    #train_datagen.fit(train_X)
    #val_datagen.fit(val_X)

    #train_flow = train_datagen.flow(train_X, train_y,
    #                               batch_size=BATCH_SIZE)
    #val_flow = val_datagen.flow(val_X, val_y,
    #                                batch_size=BATCH_SIZE)

    train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
        
    val_generator = val_datagen.flow_from_directory(
        test,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


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

    EPOCHS = 5
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 320


    result = cnn.fit_generator(
        train_generator,
        verbose=1,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_generator)
    


    #weight_path="{}_{}_{}_{}_{}_weights.best.hdf5".format(
    #    experiment, cnn_type, pretrained, feature_extraction, 'tbc'
    #)

    #checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='loss', verbose=1,
    #                            save_best_only=True, mode='min', save_weights_only = True)

    #reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

    #callbacks_list = [checkpoint]

    # training the model
      
    print()

    score = cnn.evaluate_generator(val_generator, steps=1, verbose=1)
    print()
    print('val loss:', score[0])
    print('val accuracy:', score[1])
    cnn.save('{}_{}_{}_{}_{}_img{}x{}.h5'.format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    # plot result

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

    """
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