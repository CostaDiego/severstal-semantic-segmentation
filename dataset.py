#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from os import listdir, path
import pandas as pd
import numpy as np

import cv2

class Dataset(BaseDataset):
    """ Severstel image dataset.
    Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to image folder
        masks_dir (str): path to mask folder
        colorMode (int): Color Mode
            grayscale: Load the image in grayscale mode. Interge value = 0
            RGB: Load the image in RGB mode. Integer value = 1. Default value.
            unchanged: Load the image as it is, as such including alpha channel.
                Interger value = -1
        augmentation: TODO
        preprocessing: TODO

    """

    def __init__(
            self, 
            images_dir: str,
            masks_dir: str,
            colorMode = "RGB",
            augmentation=None, 
            preprocessing=None
        ):
        self.ids = listdir(images_dir)
        self.images_fps = [path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [path.join(masks_dir, image_id) for image_id in self.ids]

        if colorMode == "RGB":
            self.colorMode = cv2.IMREAD_COLOR

        elif colorMode == "grayscale":
            self.colorMode = cv2.IMREAD_GRAYSCALE

        elif colorMode == "unchanged":
            self.colorMode = cv2.IMREAD_UNCHANGED

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def _preprocess_df(
        self,
        path: str,
        img_dir: str,
        clmn: str
        ):
        dataframe = pd.read_csv(path)
        if clmn:
            dataframe["filePath"] = dataframe[clmn].map(
                lambda x:
                    path.join(img_dir, x)
                )

        else:
            dataframe["filePath"] = dataframe[dataframe.columns[0]].map(
                lambda x:
                    path.join(img_dir, x)
                )

        dataframe["exists"] = dataframe["filePath"].map(path.exists)

        return dataframe

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i], self.colorMode)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        return len(self.ids)