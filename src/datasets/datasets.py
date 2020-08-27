from typing import Callable
import os
import numpy as np
from PIL import Image
import torch
import logging

# from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.CoarseDropout(20),
            A.Rotate(30),
            A.RandomCrop(image_size, image_size, p=0.5),
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, 0),
            A.Normalize(),
            ToTensor(),
        ]
    )


def get_test_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, 0),
            A.Normalize(),
            ToTensor(),
        ]
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        total_frames,
        root: str,
        transforms: Callable,
        with_labels: bool = True,
        anti_word: str = None,
        real_word: str = None,
    ):
        self.root = root
        self.transforms = transforms
        self.with_labels = with_labels

        self.anti_word = anti_word
        self.real_word = real_word
        self.total_frames = total_frames

    def __len__(self):
        return len(self.total_frames)

    def __getitem__(self, idx: int):
        full_path = self.total_frames[idx]

        image = Image.open(full_path)
        if self.anti_word != None:
            target = 1 if self.anti_word in full_path else 0
        elif self.real_word != None:
            target = 0 if self.real_word in full_path else 1

        image = self.transforms(image=np.array(image))["image"]

        if self.with_labels:
            return image, target
        else:
            return image
