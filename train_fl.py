import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from absl import app, flags
from sklearn.metrics import classification_report

flags.DEFINE_integer("m_div", 1, "Number of divided centers in montgomery")
flags.DEFINE_integer("c_div", 1, "Number of divided centers in china")
flags.DEFINE_string("suffix", "", "suffix")

FLAGS = flags.FLAGS

HEIGHT = 224
WIDTH = 224
CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 5
ROUNDS = 5
STEPS_PER_EPOCH = 200


def get_dataset_tbc_for_single(
    file_name,
    dataset_type="train",
    img_size=224,
    horizontal_flip=True,
    vertical_flip=False,
    random_brightness=True,
    random_contrast=True,
    random_saturation=True,
    random_hue=True,
    random_crop=True,
):
    def augment_img(img, label):
        if horizontal_flip:
            img = tf.image.random_flip_left_right(img)
        if vertical_flip:
            img = tf.image.random_flip_up_down(img)
        if random_brightness:
            img = tf.image.random_brightness(img, max_delta=0.1)
        if random_contrast:
            img = tf.image.random_contrast(img, lower=0.75, upper=1.5)
        if random_saturation:
            img = tf.image.random_saturation(img, lower=0.75, upper=1.5)
        if random_hue:
            img = tf.image.random_hue(img, max_delta=0.15)
        if random_crop:
            img = tf.image.random_crop(img, (img_size, img_size, 3))
        # Make sure the image is still in [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    npz = np.load(file_name)
    imgs, labels, n = npz["imgs"], npz["labels"], npz["n"]
    print(file_name, n)
    ds_img = tf.data.Dataset.from_tensor_slices(imgs)
    ds_label = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((ds_img, ds_label))
    if dataset_type == "train":
        ds = ds.shuffle(n, reshuffle_each_iteration=True)
        ds = ds.map(augment_img, tf.data.experimental.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.repeat()
    else:
        ds = ds.batch(n)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_model(FLAGS):
    base_model = tf.keras.applications.VGG19(
        weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    predictions = tf.keras.layers.Dense(CLASSES, activation="softmax")(x)
    for layer in base_model.layers:
        layer.trainable = False
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy("loss"),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("acc")],
    )
    return model


def main(argv):
    del argv

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    train_ds = []
    if FLAGS.m_div == 1:
        train_ds.append(get_dataset_tbc_for_single("data/train_ms.npz", dataset_type="train"))
    elif FLAGS.m_div == 2:
        train_ds.append(get_dataset_tbc_for_single("data/train_ms2_1.npz", dataset_type="train"))
        train_ds.append(get_dataset_tbc_for_single("data/train_ms2_2.npz", dataset_type="train"))

    if FLAGS.c_div == 1:
        train_ds.append(get_dataset_tbc_for_single("data/train_cs.npz", dataset_type="train"))
    elif FLAGS.c_div == 2:
        train_ds.append(get_dataset_tbc_for_single("data/train_cs2_1.npz", dataset_type="train"))
        train_ds.append(get_dataset_tbc_for_single("data/train_cs2_2.npz", dataset_type="train"))
    elif FLAGS.c_div == 3:
        train_ds.append(get_dataset_tbc_for_single("data/train_cs3_1.npz", dataset_type="train"))
        train_ds.append(get_dataset_tbc_for_single("data/train_cs3_2.npz", dataset_type="train"))
        train_ds.append(get_dataset_tbc_for_single("data/train_cs3_3.npz", dataset_type="train"))

    test_ds = get_dataset_tbc_for_single("data/test_all.npz", dataset_type="test")

    local_models = [get_model(FLAGS) for _ in train_ds]
    os.makedirs("weights/m{}c{}_{}".format(FLAGS.m_div, FLAGS.c_div, FLAGS.suffix), exist_ok=True)
    print()
    print("Number of centers:", len(train_ds))
    print("Round starts!")
    print()
    start_time = datetime.now()
    for round_num in range(1, ROUNDS + 1):
        print("Round {:2d}: ".format(round_num), end="")
        for local_model, ds in zip(local_models, train_ds):
            local_model.fit(ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=0)

        for layer_num in [-2, -1]:
            kernels = []
            biases = []
            for local_model in local_models:
                weights = local_model.layers[layer_num].get_weights()
                kernels.append(weights[0])
                biases.append(weights[1])

            new_kernel = np.average(kernels, axis=0)
            new_bias = np.average(biases, axis=0)

            for local_model in local_models:
                local_model.layers[layer_num].set_weights(np.array([new_kernel, new_bias]))

        result = local_models[0].evaluate(test_ds, verbose=0)
        print("loss: {:.4f}, acc: {:.4f},  Elasped time: {}".format(result[0], result[1], datetime.now() - start_time))

        save_to = "weights/m{}c{}_{}/r{}_loss_{:.4f}_acc_{:.4f}.h5".format(FLAGS.m_div, FLAGS.c_div, FLAGS.suffix, round_num, result[0], result[1])
        local_models[0].save(save_to)

    # evaluate
    print()
    predict_classes = []
    eval_y = []

    for x, y in test_ds:
        predict = local_models[0].predict(x)
        predict_classes.append(np.argmax(predict, axis=-1))
        eval_y.append(y)

    predict_classes = np.concatenate(predict_classes, axis=0)
    eval_y = np.concatenate(eval_y, axis=0)

    target_names = ["Class {}".format(i) for i in range(2)]
    print(classification_report(eval_y, predict_classes, target_names=target_names))


if __name__ == "__main__":
    app.run(main)
