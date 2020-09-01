from typing import Callable
import os
import numpy as np
from PIL import Image
import math
import random
import tensorflow as tf
import logging

# from facenet_pytorch import MTCNN
import albumentations as A
# from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.CoarseDropout(20),
            A.Rotate(30),
            # A.RandomCrop(image_size, image_size, p=0.5),
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, 0),
            A.Normalize(),
            # ToTensor(),
        ]
    )


def get_test_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, 0),
            A.Normalize(),
            # ToTensor(),
        ]
    )


def load_dataset(
        pathes,
        root: str,
        transforms: Callable,
        with_labels: bool = True,
        batch_size: int = 1,
        anti_word: str = None,
        real_word: str = None,
    ):

    def __load_image(path):
        image = Image.open(path).convert('RGB')
        image = transforms(image=np.array(image))["image"]
        image = tf.convert_to_tensor(image)
        return image

    def __read_label(path):
        if anti_word != None:
            target = 1 if anti_word in path else 0
        elif real_word != None:
            target = 0 if real_word in path else 1
        else:
            target = 0 if int(path.split('.')[0].split('_')[-1]) == 1 else 1
        return target

    random.shuffle(pathes)

    images = tf.data.Dataset.from_tensor_slices(list(map(__load_image, pathes))).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(list(map(__read_label, pathes))).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if with_labels:
        dataset = tf.data.Dataset.zip((images, labels))
    else:
        dataset = images
    dataset = dataset.apply(tf.data.experimental.ignore_errors()).cache().repeat().batch(batch_size, drop_remainder=False)

    return dataset
