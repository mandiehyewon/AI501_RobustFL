import tensorflow as tf
from glob import glob


def get_dataset_drd(
    target_class,
    total_epoch,
    batch_size,
    num_samples,
    img_size=512,
    dataset_path="/st2/myung/data/diabetic-retinopathy-detection/kaggle",
    balance=True,
    split="train",
    horizontal_flip=True,
    vertical_flip=False,
    random_brightness=True,
    random_contrast=True,
    random_saturation=True,
    random_hue=True,
    random_crop=True,
    crop_rate=0.8,
):
    """Params:

    """
    if random_crop:
        # If use random crop, load image with larger size
        _img_size = int(img_size / crop_rate)
    else:
        _img_size = img_size

    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (_img_size, _img_size))

        # Paths are of format ...{label}.jpeg
        _path = tf.strings.substr(path, -6, 1)
        label = tf.strings.to_number(_path)
        return img, label

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

    if target_class is None:
        if balance:
            num_samples_class = num_samples // 5
            data_list = []
            for i in range(5):
                data_list += glob("{}/{}_processed/*_{}.jpeg".format(dataset_path, split, i))[:num_samples_class]
        else:
            data_list = glob("{}/{}_processed/*.jpeg".format(dataset_path, split))[:num_samples]
    else:
        data_list = glob("{}/{}_processed/*_{}.jpeg".format(dataset_path, split, target_class))[:num_samples]

    ds = tf.data.Dataset.from_tensor_slices(data_list)
    ds = ds.shuffle(len(data_list), reshuffle_each_iteration=True)
    ds = ds.map(load_img, tf.data.experimental.AUTOTUNE)
    ds = ds.map(augment_img, tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(total_epoch)
    ds = ds.prefetch(1)
    return ds


def get_dataset_tbc(
    total_epoch_train,
    total_epoch_val,
    batch_size,
    num_samples,
    img_size=224,
    center="MontgomerySet",  # ChinaSet_AllFiles
    dataset_path="/st2/hyewon/dataset/TBc",
    balance=True,
    val_portion=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    random_brightness=True,
    random_contrast=True,
    random_saturation=True,
    random_hue=True,
    random_crop=True,
    crop_rate=0.95,
):
    """Params:

    """
    if random_crop:
        # If use random crop, load image with larger size
        _img_size = int(img_size / crop_rate)
    else:
        _img_size = img_size

    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (_img_size, _img_size))

        # Paths are of format ...{label}.png
        _path = tf.strings.substr(path, -5, 1)
        label = tf.strings.to_number(_path)
        return img, label

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

    if center is None:
        if balance:
            num_samples_class = num_samples // 2
            data_list_0 = glob("{}/MontgomerySet/CXR_png/*.png".format(dataset_path))
            data_list_1 = glob("{}/ChinaSet_AllFiles/CXR_png/*.png".format(dataset_path))
            num_val_class = int((len(data_list_0) + len(data_list_1)) * val_portion / 2)
            train_data_list = data_list_0[:num_samples_class] + data_list_1[:num_samples_class]
            val_data_list = data_list_0[num_samples_class:num_samples_class + num_val_class] \
                + data_list_1[num_samples_class:num_samples_class + num_val_class]
            tot_data_list = data_list_0 + data_list_1

        else:
            tot_data_list = glob("{}/**/CXR_png/*.png".format(dataset_path))
            num_val = int(len(tot_data_list) * val_portion)
            val_data_list = tot_data_list[:num_val]
            train_data_list = tot_data_list[num_val : num_val + num_samples]
    else:
        tot_data_list = glob("{}/{}/CXR_png/*.png".format(dataset_path, center))
        num_val = int(len(tot_data_list) * val_portion)
        val_data_list = tot_data_list[:num_val]
        train_data_list = tot_data_list[num_val : num_val + num_samples]
    print(
        "# of Data: total {}\ttrain {}\tval {}".format(
            len(tot_data_list),
            len(train_data_list),
            len(val_data_list),
        )
    )
    train_ds = tf.data.Dataset.from_tensor_slices(train_data_list)
    train_ds = train_ds.shuffle(len(train_data_list), reshuffle_each_iteration=True)
    train_ds = train_ds.map(load_img, tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(augment_img, tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.repeat(total_epoch_train)
    train_ds = train_ds.prefetch(1)

    val_ds = tf.data.Dataset.from_tensor_slices(val_data_list)
    val_ds = val_ds.map(load_img, tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.repeat(total_epoch_val)
    val_ds = val_ds.prefetch(1)

    return train_ds, val_ds



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    tf.compat.v1.enable_eager_execution()  # For test
    print("Montgomery")
    train_ds_0, val_ds_0 = get_dataset_tbc(2, 1, 8, 100)
    for img, label in train_ds_0.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_Montogomery.png")

    for img, label in val_ds_0.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("val_ds_Montogomery.png")

    print("China")
    train_ds_1, val_ds_1 = get_dataset_tbc(2, 1, 8, 100, center="ChinaSet_AllFiles")
    for img, label in train_ds_1.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_China.png")

    for img, label in val_ds_1.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("val_ds_China.png")

    print("All")
    train_ds_2, val_ds_2 = get_dataset_tbc(2, 1, 8, 100, center=None)
    for img, label in train_ds_2.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_All.png")

    for img, label in val_ds_2.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("val_ds_All.png")
    '''
    train_ds_0 = get_dataset_drd(512, target_class=0, total_epoch=1, batch_size=8)
    for img, label in train_ds_0.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_0.png")

    train_ds_4 = get_dataset_drd(512, target_class=4, total_epoch=1, batch_size=8)
    for img, label in train_ds_4.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_4.png")

    train_ds_all = get_dataset_drd(512, target_class=None, total_epoch=1, batch_size=8)
    for img, label in train_ds_all.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_all.png")
    '''