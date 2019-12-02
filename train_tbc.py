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
from art.classifiers import TensorFlowV2Classifier
from art.attacks import FastGradientMethod, CarliniLInfMethod 
# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32



# Each image's dimension is 28 x 28


# Each image's dimension is 28 x 28

# configure
experiment = "cs4"
dataset_type = "tbc"
cnn_type = "vgg19" # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"

HEIGHT = 224
WIDTH = 224
CLASSES = 2
BATCH_SIZE = 32
EPOCHS=5
STEPS_PER_EPOCH = 200
feature_extraction=False


if __name__ == "__main__":
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
        test = ms_test
    elif "cs" in experiment:
        train = cs_train
        test = cs_test
    
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
    # data prep
    """
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
    """

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
      
    restored_model1 = tf.keras.models.load_model('cs3_vgg19_imagenet_False_tbc_img224x224.h5')

    restored_model2 = tf.keras.models.load_model('ms3_vgg19_imagenet_False_tbc_img224x224.h5')
    for layer_num, layer in enumerate(cnn.layers):
      if layer_num not in [0,3,6,11,16,21,22,23]:
        layer1 = restored_model1.layers[layer_num]
        layer2 = restored_model2.layers[layer_num]
        new_weights = layer1.get_weights()[0] + layer2.get_weights()[0]
        new_bias = layer1.get_weights()[1] + layer2.get_weights()[1]
        layer.set_weights(np.array([new_weights, new_bias]))
    cnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )

    #print(cnn.summary())

    #weight_path="{}_{}_{}_{}_{}_weights.best.hdf5".format(
    #    experiment, cnn_type, pretrained, feature_extraction, 'tbc'
    #)

    #checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='loss', verbose=1,
    #                            save_best_only=True, mode='min', save_weights_only = True)

    reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='loss', factor=0.8, patience=3, verbose=1, 
                        mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001
    )

    callbacks_list = [reduceLROnPlat]

    #result = cnn.fit_generator(
     #   train_ds,
      #  verbose=1,
       # epochs=1,
        #steps_per_epoch=STEPS_PER_EPOCH,
        #validation_data=test_ds,
        #validation_steps=1,
        #callbacks=callbacks_list)

    """
    result = cnn.fit_generator(
        train_generator,
        verbose=1,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_generator)
    """

    # training the model
      
    print()

    #score = cnn.evaluate_generator(val_generator, steps=1, verbose=1)
    score = cnn.evaluate_generator(test_ds, steps=1, verbose=1)
    print()
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    cnn.save('{}_{}_{}_{}_{}_img{}x{}-divyam1.h5'.format(
        experiment, cnn_type, pretrained, feature_extraction, dataset_type, img_rows, img_cols
    ))
    # plot result

    test_y = []
    predict_classes = []
    eval_X = [] 
    eval_y = []
    for x,y in test_ds.take(1): 
      predict = cnn.predict(x)
      predict_classes.append(np.argmax(predict, axis=-1))
      test_y.append(y)
      eval_X.append(x) 
      eval_y.append(y) 

    test_y = np.concatenate(test_y, axis=0)
    predict_classes = np.concatenate(predict_classes, axis=0)
    eval_X = np.concatenate(eval_X, axis=0)
    eval_y = np.concatenate(eval_y, axis=0) 
    from sklearn.metrics import classification_report
    num_classes = 2
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(test_y, predict_classes, target_names=target_names))   

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(model=cnn, nb_classes=2, loss_object=loss_object, clip_values=(0, 1),  channel_index=3) 
    attack_fgsm = FastGradientMethod(classifier=classifier) 
    x_test_adv = attack_fgsm.generate(eval_X)

    score1 = cnn.evaluate(x_test_adv, eval_y, verbose=0)
    print("Adversarial loss:", score1[0])
    print("Adversarial accuracy:", score1[1])


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
