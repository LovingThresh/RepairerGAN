import numpy as np
import tensorflow as tf
import tf2lib as tl


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1,
                 mask=False, random_fn=True):
    # 分两种情况
    # 第一种情况：一个是针对裂缝图像与裂缝的分割标签
    # 第二种情况：一个是针对裂缝图像与非裂缝图像
    # 这两种情况还区分训练状态与非训练状态
    if not mask:
        if training:
            @tf.function
            def _map_fn(img):  # preprocessing
                if random_fn:
                    img = tf.image.random_flip_left_right(img)
                    # img = tf.image.random_brightness(img, max_delta=0.2)
                    img = tf.image.random_flip_up_down(img)
                    # img = tf.image.random_hue(img, max_delta=0.2)
                    # img = tf.image.random_contrast(img, 0.2, 0.5)
                    # img = tf.image.random_saturation(img, 5, 10)
                    # img = tf.image.random_jpeg_quality(img, 75, 95)
                img = tf.image.resize(img, [load_size, load_size])
                img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
                img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
                img = img * 2 - 1
                return img
        else:
            @tf.function
            def _map_fn(img):  # preprocessing
                img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
                img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
                img = img * 2 - 1
                return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = img[:, :, 0:1] / 255
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size,
                     training, shuffle=True, repeat=False, mask=False, random_fn=True):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = int(repeat)  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size,
                             training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat,
                             random_fn=random_fn)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size,
                             training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat, mask=mask,
                             random_fn=random_fn)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    if repeat:
        len_dataset = max(len(A_img_paths) * repeat, len(B_img_paths) * repeat) // batch_size
    else:
        len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        # 输出的训练数据一部分来自缓存池内，一部分来自新的数据
        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)
