import tensorflow as tf


def get_dataset(
    dataset_path,
    img_size,
    target_class,
    total_epoch,
    batch_size,
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
        ds = tf.data.Dataset.list_files(f"{dataset_path}/{split}_processed/*")
    else:
        ds = tf.data.Dataset.list_files(
            f"{dataset_path}/{split}_processed/*_{target_class}.jpeg"
        )
    ds = ds.repeat(total_epoch)
    ds = ds.map(load_img, tf.data.experimental.AUTOTUNE)
    ds = ds.map(augment_img, tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    tf.enable_eager_execution()  # For test
    PATH = "/st2/myung/data/diabetic-retinopathy-detection/kaggle"
    train_ds_0 = get_dataset(PATH, 512, target_class=0, total_epoch=1, batch_size=8)
    for img, label in train_ds_0.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_0.png")

    train_ds_4 = get_dataset(PATH, 512, target_class=4, total_epoch=1, batch_size=8)
    for img, label in train_ds_4.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_4.png")

    train_ds_all = get_dataset(
        PATH, 512, target_class=None, total_epoch=1, batch_size=8
    )
    for img, label in train_ds_all.take(1):
        fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
        for (x, y, ax) in zip(img, label, m_axs.flatten()):
            ax.imshow((255 * x).numpy().astype(np.uint8))
            ax.set_title("Severity {}".format(y.numpy().astype(np.uint8)))
            ax.axis("off")
        plt.savefig("train_ds_all.png")
