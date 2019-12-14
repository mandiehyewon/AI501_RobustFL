import os
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

CROP_RATE = 0.95
IMG_SIZE = 224
IMG_SIZE_CROP = int(IMG_SIZE / CROP_RATE)


def load_imgs(files, img_size):
    imgs = []
    labels = []
    for path in tqdm(files):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (img_size, img_size))
        imgs.append(img.numpy())
        labels.append(int(path[-5]))
    return imgs, labels


train_MS_files = glob("/st2/myung/data/TBc/MS/train/**/*.png")
train_CS_files = glob("/st2/myung/data/TBc/CS/train/**/*.png")
test_MS_files = glob("/st2/myung/data/TBc/MS/test/**/*.png")
test_CS_files = glob("/st2/myung/data/TBc/CS/test/**/*.png")

np.random.shuffle(train_MS_files)
np.random.shuffle(train_CS_files)

n_ms = len(train_MS_files)
n_cs = len(train_CS_files)
n_ms_2 = n_ms // 2
n_cs_2 = n_cs // 2
n_cs_3 = n_cs // 3

img_train_ms, label_train_ms = load_imgs(train_MS_files, IMG_SIZE_CROP)
img_train_cs, label_train_cs = load_imgs(train_CS_files, IMG_SIZE_CROP)
img_test_ms, label_test_ms = load_imgs(test_MS_files, IMG_SIZE)
img_test_cs, label_test_cs = load_imgs(test_CS_files, IMG_SIZE)

img_train_all = img_train_ms + img_train_cs
label_train_all = label_train_ms + label_train_cs
img_test_all = img_test_ms + img_test_cs
label_test_all = label_test_ms + label_test_cs

os.makedirs("data", exist_ok=True)
np.savez("data/train_all.npz", imgs=img_train_all, labels=label_train_all, n=len(label_train_all))
np.savez("data/test_all.npz", imgs=img_test_all, labels=label_test_all, n=len(label_test_all))

np.savez("data/test_ms.npz", imgs=img_test_ms, labels=label_test_ms, n=len(label_test_ms))
np.savez("data/test_cs.npz", imgs=img_test_cs, labels=label_test_cs, n=len(label_test_cs))

np.savez("data/train_ms.npz", imgs=img_train_ms, labels=label_train_ms, n=len(label_train_ms))
np.savez(
    "data/train_ms2_1.npz",
    imgs=img_train_ms[:n_ms_2],
    labels=label_train_ms[:n_ms_2],
    n=len(label_train_ms[:n_ms_2]),
)
np.savez(
    "data/train_ms2_2.npz",
    imgs=img_train_ms[n_ms_2:],
    labels=label_train_ms[n_ms_2:],
    n=len(label_train_ms[n_ms_2:]),
)

np.savez("data/train_cs.npz", imgs=img_train_cs, labels=label_train_cs, n=len(label_train_cs))
np.savez(
    "data/train_cs2_1.npz",
    imgs=img_train_cs[:n_cs_2],
    labels=label_train_cs[:n_cs_2],
    n=len(label_train_cs[:n_cs_2]),
)
np.savez(
    "data/train_cs2_2.npz",
    imgs=img_train_cs[n_cs_2:],
    labels=label_train_cs[n_cs_2:],
    n=len(label_train_cs[n_cs_2:]),
)
np.savez(
    "data/train_cs3_1.npz",
    imgs=img_train_cs[:n_cs_3],
    labels=label_train_cs[:n_cs_3],
    n=len(label_train_cs[:n_cs_3]),
)
np.savez(
    "data/train_cs3_2.npz",
    imgs=img_train_cs[n_cs_3 : 2 * n_cs_3],
    labels=label_train_cs[n_cs_3 : 2 * n_cs_3],
    n=len(label_train_cs[n_cs_3 : 2 * n_cs_3]),
)
np.savez(
    "data/train_cs3_3.npz",
    imgs=img_train_cs[2 * n_cs_3 :],
    labels=label_train_cs[2 * n_cs_3 :],
    n=len(label_train_cs[2 * n_cs_3 :]),
)
