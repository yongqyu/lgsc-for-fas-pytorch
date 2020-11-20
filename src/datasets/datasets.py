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


# @tf.function
def augment(image):
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, max_delta=0.1) # Random brightness
    image = tf.image.random_contrast(image, 0.8, 1.1)
    image = tf.image.random_jpeg_quality(image, 75, 100)
    return image

def hard_augment(image):
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness
    image = tf.image.random_contrast(image, 0.7, 1.2)
    image = tf.image.random_jpeg_quality(image, 75, 100)
    return image

# @tf.function
def normalize(image):
    max_pixel_value=255.0
    image = tf.divide(image, max_pixel_value)
    mean=[[[0.485, 0.456, 0.406]]]
    std=[[[0.229, 0.224, 0.225]]]
    image = tf.divide(tf.math.subtract(image, tf.constant(mean, dtype=tf.float32)), tf.constant(std, dtype=tf.float32))
    return image

# @tf.function
def get_train_augmentations(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = normalize(image)
    image = augment(image)
    return image

# @tf.function
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
    images = tf.data.Dataset.from_tensor_slices(pathes)#.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    images = images.map(transforms, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if with_labels:
        labels = [*map(__read_label, pathes)]
        labels = tf.data.Dataset.from_tensor_slices(labels)#.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # pathes = tf.data.Dataset.from_tensor_slices(pathes)#.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((images, labels))
    else:
        dataset = images


    return dataset


class Hardload_Generator(tf.keras.utils.Sequence) :

    def __init__(self,
        data,
        heavy_data,
        transforms: Callable,
        batch_size: int,
        with_labels: bool = True,
    ):
        self.dir_max_sample_cnt = 2
        #real_data = [(x,y) for x,y in data if y == 0]
        #fake_data = random.sample([(x,y) for x,y in data if y == 1], len(real_data))
        #data = fake_data+real_data
        self.sample_data = data
        self.heavy_data = heavy_data
        self.data = [x for dir_list in self.sample_data for x in random.sample(dir_list, min(len(dir_list),self.dir_max_sample_cnt))] + self.heavy_data
        random.shuffle(self.data)
        self.pathes, self.labels = zip(*self.data)
        self.transforms = transforms
        self.batch_size = batch_size

        self.with_labels = with_labels

    def __len__(self) :
        return (np.ceil(len(self.pathes) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        batch_pathes = self.pathes[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        # batch_pathes = tf.convert_to_tensor(batch_x)
        images = map(self.transforms, batch_pathes) #num_parallel_calls = tf.data.experimental.AUTOTUNE)
        images = tf.stack(list(images), axis=0)
        images = tfa.image.random_cutout(images, 200)

        if self.with_labels:
            labels = tf.convert_to_tensor(np.array(batch_labels))#.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            # pathes = tf.convert_to_tensor(np.array(batch_pathes))#.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dataset = (images, labels)
        else:
            dataset = images

        return dataset

    def on_epoch_end(self):
        self.data = [x for dir_list in self.sample_data for x in random.sample(dir_list, min(len(dir_list),self.dir_max_sample_cnt))] + self.heavy_data
        random.shuffle(self.data)
        self.pathes, self.labels = zip(*self.data)
