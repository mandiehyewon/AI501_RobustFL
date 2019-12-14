import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from absl import app, flags
from sklearn.metrics import classification_report


from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, CarliniLInfMethod, ProjectedGradientDescent, CarliniL2Method

flags.DEFINE_integer("m_div", 1, "Number of divided centers in montgomery")
flags.DEFINE_integer("c_div", 1, "Number of divided centers in china")
flags.DEFINE_string("suffix", "", "suffix")

FLAGS = flags.FLAGS

HEIGHT = 224
WIDTH = 224
CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 5
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


def get_adv_dataset_tbc_for_single(
    file_name,
    attack_type,
    classifier,
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
    ds = ds.map(augment_img, tf.data.experimental.AUTOTUNE)

    imgs_aug = []
    labels_aug = []
    for img, label in ds:
        imgs_aug.append(img.numpy())
        labels_aug.append(label.numpy())
    imgs_aug = np.array(imgs_aug)
    labels_aug = np.array(labels_aug)

    imgs_adv = generate_attack(attack_type, classifier, imgs_aug, labels_aug)
    ds_img_adv = tf.data.Dataset.from_tensor_slices(imgs_adv)
    ds_label_adv = tf.data.Dataset.from_tensor_slices(labels_aug)
    ds_adv = tf.data.Dataset.zip((ds_img_adv, ds_label_adv))
    ds = ds.concatenate(ds_adv)

    ds = ds.shuffle(2*n, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.repeat()
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

def eval_precision_recall(model, ds):
    predict_classes = []
    eval_y = []

    for x, y in ds:
        predict = model.predict(x)
        predict_classes.append(np.argmax(predict, axis=-1))
        eval_y.append(y)

    predict_classes = np.concatenate(predict_classes, axis=0)
    eval_y = np.concatenate(eval_y, axis=0)

    print(classification_report(eval_y, predict_classes, target_names=["Class 0", "Class 1"]))

def generate_attack(attack_type, classifier, X, y):
    if attack_type == "fgm":
        attack = FastGradientMethod(classifier, eps=0.03)
        X_adv = attack.generate(x=X)
    elif attack_type == "carlin":
        attack = CarliniLInfMethod(classifier, eps=0.03, max_iter=100, learning_rate=0.01)
        X_adv = attack.generate(x=X, y=tf.keras.utils.to_categorical(np.ones_like(y)))
    elif attack_type == "pgd_inf":
        attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.03, eps_step=0.007,max_iter=100)
        X_adv = attack.generate(x=X)
    elif attack_type == "pgd_1":
        attack = ProjectedGradientDescent(classifier, norm=1, eps=12, eps_step=0.1,max_iter=100)
        X_adv = attack.generate(x=X)
    elif attack_type == "pgd_2":
        attack = ProjectedGradientDescent(classifier, norm=2, eps=0.5, eps_step=0.05,max_iter=100)
        X_adv = attack.generate(x=X)
    
    perturbation = np.mean(np.abs((X - X_adv)))
    print('Average perturbation: {:.10f}'.format(perturbation))
    return X_adv

def main(argv):
    del argv

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    test_ds = get_dataset_tbc_for_single("data/test_all.npz", dataset_type="test")

    pretrained_fl = "weights/m1c1_2/r5_loss_0.4604_acc_0.8519.h5"
    local_models = [tf.keras.models.load_model(pretrained_fl) for _ in range(FLAGS.m_div + FLAGS.c_div)]
    os.makedirs("weights/m{}c{}_{}".format(FLAGS.m_div, FLAGS.c_div, FLAGS.suffix), exist_ok=True)

    # ------------------      
    # Adversarial Attack
    # ------------------
    # for x, y in test_ds:
    #     x_test = x.numpy()
    #     y_test = y.numpy()

    # robust_model = KerasClassifier(model=local_models[0], clip_values=(0,1))

    # for attack_type in ["fgm", "carlin", "pgd_inf", "pgd_1", "pgd_2"]:
    #     X_adv = generate_attack(attack_type, robust_model, x_test, y_test)

    #     predict = robust_model.predict(X_adv)
    #     predict_classes = np.argmax(predict, axis=-1)
    #     target_names = ["Class {}".format(i) for i in range(CLASSES)]
    #     print(classification_report(y_test, predict_classes, target_names=target_names))
    #     accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
    #     print('Accuracy on {} Method test examples: {:.3f}%'.format(attack_type.upper(), accuracy * 100))
    #     print()

    # ------------------      
    # Get Adversarial data
    # ------------------
    train_ds = []
    robust_model = KerasClassifier(model=local_models[0], clip_values=(0,1))
    if FLAGS.m_div == 1:
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_ms.npz", "pgd_inf", robust_model))
    elif FLAGS.m_div == 2:
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_ms2_1.npz", "pgd_inf", robust_model))
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_ms2_2.npz", "pgd_inf", robust_model))
    
    if FLAGS.c_div == 1:
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_cs.npz", "pgd_inf", robust_model))
    elif FLAGS.c_div == 2:
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_cs2_1.npz", "pgd_inf", robust_model))
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_cs2_2.npz", "pgd_inf", robust_model))
    elif FLAGS.c_div == 3:
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_cs3_1.npz", "pgd_inf", robust_model))
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_cs3_2.npz", "pgd_inf", robust_model))
        train_ds.append(get_adv_dataset_tbc_for_single("data/train_cs3_3.npz", "pgd_inf", robust_model))

    # ------------------      
    # Federated Learning
    # ------------------      
    print()
    print("Number of centers:", len(train_ds))
    print("Round starts!")
    print()
    start_time = datetime.now()
    for round_num in range(1, 5 + 1):
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
        print('loss: {:.4f}, acc: {:.4f},  Elasped time: {}'.format(result[0], result[1], datetime.now() - start_time))

        save_to = "weights/m{}c{}_{}/r{}_loss_{:.4f}_acc_{:.4f}.h5".format(FLAGS.m_div, FLAGS.c_div, FLAGS.suffix, round_num, result[0], result[1])
        local_models[0].save(save_to)

    eval_precision_recall(local_models[0], test_ds)
    
    # ------------------      
    # Adversarial Testing
    # ------------------  
    for x, y in test_ds:
        x_test = x.numpy()
        y_test = y.numpy()
    
    robust_model = KerasClassifier(model=local_models[0], clip_values=(0,1))

    for attack_type in ["fgm", "carlin", "pgd_inf", "pgd_1", "pgd_2"]:
        X_adv = generate_attack(attack_type, robust_model, x_test, y_test)

        predict = robust_model.predict(X_adv)
        predict_classes = np.argmax(predict, axis=-1)
        target_names = ["Class {}".format(i) for i in range(CLASSES)]
        print(classification_report(y_test, predict_classes, target_names=target_names))
        accuracy = np.sum(np.argmax(predict, axis=1) == y_test) / len(y_test)
        print('Accuracy on {} Method test examples: {:.3f}%'.format(attack_type.upper(), accuracy * 100))
        print()


if __name__ == "__main__":
    app.run(main)
