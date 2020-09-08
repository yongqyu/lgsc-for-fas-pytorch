from typing import Callable
import os
import numpy as np
from PIL import Image
import math
import random
import tensorflow as tf
import tensorflow_addons as tfa
import logging

import albumentations as A
# from albumentations.pytorch import ToTensorV2 as ToTensor

'''
def get_train_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.CoarseDropout(20),
            A.Rotate(30),
            A.RandomCrop(image_size, image_size, p=0.5),
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
'''

@tf.function
def augment(image):
    # image = tfa.image.rotate(image, tf.constant(np.pi/8))#tf.constant(np.random.uniform(high=0.5)))
    image = tf.image.random_crop(image, (224, 224, 3))
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, 0.5)
    return image

@tf.function
def normalize(image):
    image = (image - 127.5)/128.0
    return image

@tf.function
def get_train_augmentations(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = augment(image)
    return image

@tf.function
def get_test_augmentations(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    return image

def load_dataset(
        pathes,
        transforms: Callable,
        with_labels: bool = True,
        anti_word: str = None,
        real_word: str = None,
    ):

    def __read_label(path):
        if anti_word != None:
            target = 1 if anti_word in path else 0
        elif real_word != None:
            target = 0 if real_word in path else 1
        else:
            target = 0 if int(path.split('.')[0].split('_')[-1]) == 1 else 1
        return target

    # images = [*map(__load_image, pathes)]
    # images = tf.data.Dataset.from_tensor_slices(images)
    images = tf.data.Dataset.from_tensor_slices(pathes).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    images = images.map(transforms, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if with_labels:
        labels = [*map(__read_label, pathes)]
        labels = tf.data.Dataset.from_tensor_slices(labels).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        pathes = tf.data.Dataset.from_tensor_slices(pathes).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((images, labels, pathes))
    else:
        dataset = images


    return dataset
