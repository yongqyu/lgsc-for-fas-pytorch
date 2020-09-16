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


@tf.function
def augment(image):
    image = tf.image.random_crop(image, (224, 224, 3))
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_saturation(image, 0.5, 2.0)
    # image = tf.image.random_brightness(image, 0.5)
    return image

@tf.function
def normalize(image):
    max_pixel_value=255.0
    image = tf.divide(image, max_pixel_value)
    mean=[[[0.485, 0.456, 0.406]]]
    std=[[[0.229, 0.224, 0.225]]]
    image = tf.divide(tf.math.subtract(image, tf.constant(mean, dtype=tf.float32)), tf.constant(std, dtype=tf.float32))
    return image

@tf.function
def get_train_augmentations(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = normalize(image)
    image = augment(image)
    return image

@tf.function
def get_test_augmentations(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = normalize(image)
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
