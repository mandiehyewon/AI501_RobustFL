import tensorflow as tf


def get_dataset(dataset_path, img_size, target_class, split="train"):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, img_size)
        return img

    ds_list = tf.data.Dataset.list_files(
        f"{dataset_path}/{split}/*_{target_class}.jpeg"
    )
    ds = ds_list.interleave(load_img)

    return ds
