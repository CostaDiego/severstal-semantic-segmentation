import pandas as pd
from PIL import Image
import numpy as np

import matplotlib.image as plt
import os
from pathlib import Path

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 256

def generateMask(dataFrame, maskDir = "masksDir", trainDir = None):
    if trainDir:
        trainPath = Path(trainDir)
        # baseFolder = trainDir
        imageIds = []

        for ids in trainPath.iterdir():
            imageIds.append(os.path.split(str(ids))[1])

    if not os.path.exists(maskDir):
        os.mkdir(maskDir)
    
    img_df = pd.read_csv(dataFrame)
    uniqueIds = img_df["ImageId"].unique()
    
    for ids in uniqueIds:
        sml_df = img_df[img_df['ImageId'] == str(ids)]
        values = sml_df.values

        mask_img = np.zeros((IMAGE_HEIGHT*IMAGE_WIDTH,1), dtype=int)
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

            mask_img[pxl_mask] = clss

        mask = np.reshape(mask_img, (IMAGE_WIDTH,IMAGE_HEIGHT)).T

        savePath = os.path.join(maskDir, mask_name)

        plt.imsave(savePath, mask)

        if trainDir:
            imageIds.remove(mask_name)

    if trainDir:
        overwrited_msks = 0
        for ids in imageIds:
            mask_img = np.zeros((IMAGE_HEIGHT*IMAGE_WIDTH,1), dtype=int)
            mask = np.reshape(mask_img, (IMAGE_WIDTH,IMAGE_HEIGHT)).T
            savePath = os.path.join(maskDir, ids)

            if not os.path.isfile(savePath):
                plt.imsave(savePath, mask)
            else:
                overwrited_msks += 1

    print(
        f"The script ended with {overwrited_msks}",
        "attempts to overwrite a preexistent mask"
    )
    