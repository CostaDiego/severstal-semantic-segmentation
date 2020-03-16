import pandas as pd
import numpy as np

import cv2
import matplotlib.pyplot as pyplot

import os
from pathlib import Path
import shutil

_IMAGE_WIDTH = 1600
_IMAGE_HEIGHT = 256

def generateMask(dataFrame, maskDir = "masksDir", trainDir = None):
    if trainDir:
        trainPath = Path(trainDir)
        imageIds = []

        for ids in trainPath.iterdir():
            imageIds.append(os.path.split(str(ids))[1])
        
        for ids in imageIds:
            if not str(os.path.splitext(str(ids))[1]) ==".jpg":
                imageIds.remove(ids)

    if not os.path.exists(maskDir):
        os.mkdir(maskDir)
    
    img_df = pd.read_csv(dataFrame)
    uniqueIds = img_df["ImageId"].unique()
    
    for ids in uniqueIds:
        sml_df = img_df[img_df['ImageId'] == str(ids)]
        values = sml_df.values

        mask_img = np.zeros((_IMAGE_HEIGHT*_IMAGE_WIDTH,1), dtype=int)
        mask_name = None

        for key in range(len(values)):
            img_name, clss, enc_pxl = values[key][0],values[key][1],values[key][2]

            mask_name = str(img_name)
            clss = int(clss)
            enc_pxl = enc_pxl.split()
            enc_pxl = [int(i) for i in enc_pxl]

            pxl, pxl_ct = [],[]

            for i in range(len(enc_pxl)):
                if i%2 == 0:
                    pxl.append(enc_pxl[i])
                else:
                    pxl_ct.append(enc_pxl[i])
            
            concat_pxl = [
                list(range(pxl[i], pxl[i] + pxl_ct[i])) for i in range(len(pxl))
            ]

            pxl_mask = sum(concat_pxl,[])

            # mask_img[pxl_mask] = clss
            for i in pxl_mask:
                mask_img[i-1] = clss

        mask = np.reshape(mask_img, (_IMAGE_WIDTH,_IMAGE_HEIGHT)).T

        savePath = os.path.join(maskDir, mask_name)

        cv2.imwrite(savePath, mask)

        if trainDir:
            imageIds.remove(mask_name)

    if trainDir:
        overwrited_msks = 0
        for ids in imageIds:
            mask_img = np.zeros((_IMAGE_HEIGHT*_IMAGE_WIDTH,1), dtype=int)
            mask = np.reshape(mask_img, (_IMAGE_WIDTH,_IMAGE_HEIGHT)).T
            savePath = os.path.join(maskDir, ids)

            if not os.path.isfile(savePath):
                cv2.imwrite(savePath, mask)
            else:
                overwrited_msks += 1

    print(
        f"The script ended with {overwrited_msks}",
        "attempts to overwrite a preexistent mask"
    )

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    pyplot.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        pyplot.subplot(1, n, i + 1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.title(' '.join(name.split('_')).title())
        pyplot.imshow(image)
    pyplot.show()

def validadeFolder(
    path: str,
    renew: bool = False):
    path = str(path)

    if os.path.isdir(path) and renew:
        shutil.rmtree(path)
        os.mkdir(path)
        return True
    
    elif os.path.isdir(path) and not renew:
        return False

    elif not os.path.isdir(path):
        os.mkdir(path)
        return True
    