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
import numpy as np

parser = argparse.ArgumentParser(description="Dataset creation")  

parser.add_argument("--yamaha_data_path", type=str,
                    help="path for the new yamaha dataset", default = "yd/")

parser.add_argument("--frame_rate", type=float,
                    help="framerate to capture image for every n seconds",
                    default=0.1)

opts = parser.parse_args()

DIM=(1280, 720)
K=np.array([[406.9505224854568, 0.0, 661.4506493463073], [0.0, 407.839666443687, 360.787400388333], [0.0, 0.0, 1.0]])
D=np.array([[-0.011205156573322632], [0.007034153924762964], [-0.007526951878842564], [0.0016083473413905825]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

def getFrame(sec, folder):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        img_name = str(count).zfill(6)
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(opts.yamaha_data_path + os.sep + folder + os.sep + folder + "_"+img_name+".png", image)
    return hasFrames


for file in glob.glob("*.avi"):
    i = 0
    ignore_count = 2 ## ignore first 2 captured images
    vidcap = cv2.VideoCapture(file)
    a = file.replace(".", "_").split("_")[1]
    folder = '-'.join(a.split("-")[-3:])

    print(folder)
    Path(opts.yamaha_data_path + os.sep + folder).mkdir(parents=True, exist_ok=True)
    sec = 0
    count=0
    success = getFrame(sec, folder)
    while success:
        ## problem reading these videos. necessary to avoid infinite loop
        #if (folder =="08-53-36"):
            #break
            
        if i>=ignore_count:
            count = count + 1
        i += 1
        sec = sec + opts.frame_rate
        #print(count)
        sec = round(sec, 2)
        success = getFrame(sec, folder)

