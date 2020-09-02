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

@tf.function
def augment(image):
    image = tf.keras.preprocessing.image.random_rotation(image, 20)
    image = tf.image.random_crop(image, (224, 224, 3))
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, 0.5)
    return image

@tf.function
def normalize(image):
    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image))
    image = (2 * image) - 1
    return image

def load_dataset(
        pathes,
        root: str,
        transforms: Callable,
        with_labels: bool = True,
        batch_size: int = 1,
        anti_word: str = None,
        real_word: str = None,
    ):

    @tf.function
    def __load_image(path):
        # image = Image.open(path).convert('RGB')
        # image = transforms(image=np.array(image))["image"]
        # image = tf.convert_to_tensor(image)
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels = 3, dtype = tf.float32)
        image = augment(image)
        image = normalize(image)
        return image

    def __read_label(path):
        if anti_word != None:
            target = 1 if anti_word in path else 0
        elif real_word != None:
            target = 0 if real_word in path else 1
        else:
            target = 0 if int(path.split('.')[0].split('_')[-1]) == 1 else 1
        return target

    def preprocess(path):
        image = __load_image(path)
        if with_labels:
            label = __read_label(path)
            return image, label
        else:
            return image

    # images = [*map(__load_image, pathes)]
    # images = tf.data.Dataset.from_tensor_slices(images)
    images = tf.data.Dataset.from_tensor_slices(pathes)
    images = images.map(__load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if with_labels:
        labels = [*map(__read_label, pathes)]
        labels = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((images, labels))
    else:
        dataset = images

    dataset = dataset.shuffle(buffer_size = 1024)
    dataset = dataset.batch(batch_size = batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE).cache()

    return dataset
