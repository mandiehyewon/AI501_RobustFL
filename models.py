import tensorflow as tf

# inception_v3 : image size needs to be at least 75x75
# others works in image size 32x32

def create_compiled_keras_cnn1_model(height, width, classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(height, width, 3)
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_cnn3_model(height, width, classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(height, width, 3)
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
            tf.keras.layers.Dense(classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_cnn4_model(height, width, classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(height, width, 3)
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
            tf.keras.layers.Dense(classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def create_compiled_keras_vgg16_model(height, width, classes):
    base_model = tf.keras.applications.VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=(height, width, 3))

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = tf.keras.layers.Dense(classes, activation='softmax')(x)
    cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    cnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return cnn

def create_compiled_keras_vgg19_model(height, width, classes):
    base_model = tf.keras.applications.VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=(height, width, 3))

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = tf.keras.layers.Dense(classes, activation='softmax')(x)
    cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    cnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )


    return cnn



def create_compiled_keras_resnet50_model(height, width, classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet',
                    include_top=False,
                    input_shape=(height, width, 3))

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = tf.keras.layers.Dense(classes, activation='softmax')(x)
    cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    cnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )


    return cnn


def create_compiled_keras_inception_v3_model(height, width, classes):
    base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                    include_top=False,
                    input_shape=(height, width, 3))

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = tf.keras.layers.Dense(classes, activation='softmax')(x)
    cnn = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    cnn.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return cnn


def get_model(FLAGS):
    if FLAGS.cnn_type == "cnn1":
        cnn = create_compiled_keras_cnn1_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)
    elif FLAGS.cnn_type == "cnn3":
        cnn = create_compiled_keras_cnn3_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)
    elif FLAGS.cnn_type == "cnn4":
        cnn = create_compiled_keras_cnn4_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)
    elif FLAGS.cnn_type == "vgg16":
        cnn = create_compiled_keras_vgg16_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)
    elif FLAGS.cnn_type == "vgg19":
        cnn = create_compiled_keras_vgg19_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)
    elif FLAGS.cnn_type == "resnet50":
        cnn = create_compiled_keras_resnet50_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)
    elif FLAGS.cnn_type == "inception_v3":
        cnn = create_compiled_keras_inception_v3_model(FLAGS.height, FLAGS.width, FLAGS.num_classes)

    return cnn
