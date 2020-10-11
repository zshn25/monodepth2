# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:20:53 2020

@author: chand
"""

import cv2
import os
import glob
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Dataset creation")  

parser.add_argument("--yamaha_data_path", type=str, required=True,
                    help="path for the new yamaha dataset")

parser.add_argument("--frame_rate", type=int,
                    help="framerate to capture image for every n seconds",
                    default=1)

opts = parser.parse_args()

def getFrame(sec, folder):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        img_name = str(count).zfill(6)
        cv2.imwrite(dataset_path + os.sep + folder + os.sep + folder + "_"+img_name+".png", image)
    return hasFrames


for file in glob.glob("*.avi"):
    i = 0
    ignore_count = 2 ## ignore first 2 captured images
    vidcap = cv2.VideoCapture(file)
    a = file.replace(".", "_").split("_")[1]
    folder = '-'.join(a.split("-")[-3:])

    print(folder)
    Path(dataset_path + os.sep + folder).mkdir(parents=True, exist_ok=True)
    sec = 0
    count=0
    success = getFrame(sec, folder)
    while success:
        ## problem reading these videos. necessary to avoid infinite loop
        if (folder =="08-53-36" and count==466):
            break
        if (folder =="08-53-19" and count==370):
            break
        if i>=ignore_count:
            count = count + 1
        i += 1
        sec = sec + frameRate
        #print(count)
        sec = round(sec, 2)
        success = getFrame(sec, folder)

