import tensorflow as tf

# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32

# Each image's dimension is 28 x 28
raw_img_rows, raw_img_cols = 28, 28
img_channels = 3
min_img_rows, min_img_cols = 224, 224
img_rows = max(min_img_rows, raw_img_rows)
img_cols = max(min_img_cols, raw_img_cols)
input_shape = (img_rows, img_cols, img_channels)
pretrained = None # "imagenet" or None
transfer_learning = False # True or False (False: pretrained weight is fixed)

"""
# configure
data_dir = "/st2/myung/data/"
dataset_type = "fashion-mnist"
cnn_type = "cnn4" # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"
pretrained = "imagenet" # "imagenet" or None
transfer_learning = False # True or False (False: pretrained weight is fixed)
random_seed = 42
"""


def create_compiled_keras_cnn1_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_cnn3_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_cnn4_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_vgg16_model():
    vgg16_conv = tf.keras.applications.VGG16(
        weights=pretrained, include_top=False, input_shape=input_shape
    )
    vgg16_conv.summary()
    if not transfer_learning and pretrained != None:
        vgg16_conv.trainable = False
    model = tf.keras.models.Sequential(
        [
            vgg16_conv,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_vgg19_model():

    vgg19_conv = tf.keras.applications.VGG19(
        weights=pretrained, include_top=False, input_shape=input_shape
    )
    if not transfer_learning and pretrained != None:
        vgg19_conv.trainable = False
    model = tf.keras.models.Sequential(
        [
            vgg19_conv,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_resnet50_model():
    resnet50_conv = tf.keras.applications.ResNet50(
        weights=pretrained, include_top=False, input_shape=input_shape
    )
    if not transfer_learning and pretrained != None:
        resnet50_conv.trainable = False
    model = tf.keras.models.Sequential(
        [
            resnet50_conv,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_inception_v3_model():

    inception_v3_conv = tf.keras.applications.InceptionV3(
        weights=pretrained, include_top=False, input_shape=input_shape
    )
    if not transfer_learning and pretrained != None:
        inception_v3_conv.trainable = False
    model = tf.keras.models.Sequential(
        [
            inception_v3_conv,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def get_model(FLAGS):
    if FLAGS.cnn_type == "cnn1":
        cnn_model = create_compiled_keras_cnn1_model()
    elif FLAGS.cnn_type == "cnn3":
        cnn_model = create_compiled_keras_cnn3_model()
    elif FLAGS.cnn_type == "cnn4":
        cnn_model = create_compiled_keras_cnn4_model()
    elif FLAGS.cnn_type == "vgg16":
        cnn_model = create_compiled_keras_vgg16_model()
    elif FLAGS.cnn_type == "vgg19":
        cnn_model = create_compiled_keras_vgg19_model()
    elif FLAGS.cnn_type == "resnet50":
        cnn_model = create_compiled_keras_resnet50_model()
    elif FLAGS.cnn_type == "inception_v3":
        cnn_model = create_compiled_keras_inception_v3_model()

    return cnn_model


if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
    import matplotlib.pyplot as plt
    import os

    data_dir = "/st2/myung/data/"
    dataset_type = "fashion-mnist"
    cnn_type = "cnn4"  # "cnn1","cnn3","cnn4","vgg16","vgg19","resnet50","inception_v3"
    pretrained = "imagenet"  # "imagenet" or None
    transfer_learning = False  # True or False (False: pretrained weight is fixed)
    random_seed = 42

    tf.compat.v1.set_random_seed(random_seed)

    # Load training and test data into dataframes
    data_train = pd.read_csv(os.path.join(data_dir, dataset_type, "train.csv"))
    data_test = pd.read_csv(os.path.join(data_dir, dataset_type, "test.csv"))

    # X forms the training images, and y forms the training labels
    X = np.array(data_train.iloc[:, 1:])
    y = np.array(data_train.iloc[:, 0], dtype=np.int32)

    # X_test forms the test images, and y_test forms the test labels
    X_test = np.array(data_test.iloc[:, 1:])
    y_test = np.array(data_test.iloc[:, 0], dtype=np.int32)

    # Display their new shapes
    print(X.shape, X_test.shape)

    # Convert the training and test images into 3 channels
    X = np.dstack([X] * 3)
    X_test = np.dstack([X_test] * 3)

    # Display their new shapes
    print(X.shape, X_test.shape)

    X = X.reshape(X.shape[0], raw_img_rows, raw_img_cols, img_channels)
    X_test = X_test.reshape(X_test.shape[0], raw_img_rows, raw_img_cols, img_channels)

    if raw_img_cols != img_cols or raw_img_cols != img_cols:
        X = np.asarray(
            [
                img_to_array(array_to_img(im, scale=False).resize((img_rows, img_cols)))
                for im in X
            ]
        )
        X_test = np.asarray(
            [
                img_to_array(array_to_img(im, scale=False).resize((img_rows, img_cols)))
                for im in X_test
            ]
        )

    # Prepare the training images
    X = X.astype("float32")
    X /= 255

    # Prepare the test images
    X_test = X_test.astype("float32")
    X_test /= 255

    if raw_img_cols != img_cols or raw_img_cols != img_cols:
        plt.imshow(X[0].reshape(img_rows, img_cols, img_channels), cmap="gray")
        plt.grid(False)
        plt.savefig("rescaled_img{}x{}_dataset_example.png".format(img_rows, img_cols))
        plt.close()
    # Here I split original training data to sub-training (80%) and validation data (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=13
    )
    print(y_val)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

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
    result = cnn.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=10,
        verbose=1,
        validation_data=(X_val, y_val),
    )

    score = cnn.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    cnn.save(
        "{}_{}_{}_{}_img{}x{}.h5".format(
            cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
        )
    )

    # plot result

    accuracy = result.history["acc"]
    val_accuracy = result.history["val_acc"]
    loss = result.history["loss"]
    val_loss = result.history["val_loss"]
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, "red", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "blue", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig(
        "{}_{}_{}_{}_img{}x{}_accuracy.pdf".format(
            cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
        )
    )
    plt.close()

    plt.plot(epochs, loss, "red", label="Training loss")
    plt.plot(epochs, val_loss, "blue", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(
        "{}_{}_{}_{}_img{}x{}_loss.pdf".format(
            cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
        )
    )
    plt.close()

    # classification report
    predicted_classes = cnn.predict_classes(X_test)

    y_true = data_test.iloc[:, 0]
    correct = (predicted_classes == y_true).to_numpy().nonzero()[0]
    incorrect = (predicted_classes != y_true).to_numpy().nonzero()[0]

    from sklearn.metrics import classification_report

    target_names = ["Class {}".format(i) for i in range(10)]
    print(classification_report(y_true, predicted_classes, target_names=target_names))

    # plot correctly predicted classes
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(
            X_test[correct].reshape(img_rows, img_cols, img_channels),
            cmap="gray",
            interpolation="none",
        )
        plt.title(
            "Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct])
        )
        plt.tight_layout()
    plt.savefig(
        "{}_{}_{}_{}_img{}x{}_correct_prediction.pdf".format(
            cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
        )
    )
    plt.close()

    # plot incorrectly predicted classes
    for i, incorrect in enumerate(incorrect[0:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(
            X_test[incorrect].reshape(img_rows, img_cols, img_channels),
            cmap="gray",
            interpolation="none",
        )
        plt.title(
            "Predicted {}, Class {}".format(
                predicted_classes[incorrect], y_true[incorrect]
            )
        )
        plt.tight_layout()
    plt.savefig(
        "{}_{}_{}_{}_img{}x{}_incorrect_prediction.pdf".format(
            cnn_type, pretrained, transfer_learning, dataset_type, img_rows, img_cols
        )
    )
    plt.close()
