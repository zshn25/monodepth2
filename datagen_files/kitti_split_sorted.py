# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:28:05 2020

@author: chand
"""

### Place it in videos folder

import re
import os
import argparse
import platform
from glob import glob

file_dir = os.path.dirname(__file__)
splits_path = os.path.join(file_dir, "splits")

parser = argparse.ArgumentParser(description="Split options")  

## Zerobike dataset
parser.add_argument("--zerobike_data_path", type=str,
                    help="path to the zerobike training data",
                    default="/home/e2r/Desktop/e2r/data/kitti-raw/")
                    #default="zerobike_dataset")

parser.add_argument("--zerobike_split_path", type=str,
                    help="path to the zerobike dataset split file",
                    default=os.path.join(splits_path, "kitti_split_sorted"))

parser.add_argument("--gen_zerobike_split", 
                    help="generate zerobike dataset split",
                    action="store_true", default = True)


opts = parser.parse_args()

    

train_path = os.path.join(opts.zerobike_data_path)
val_path = os.path.join(opts.zerobike_data_path)

## writing zerobike train files
train_file = open(opts.zerobike_split_path + os.sep + "train_files.txt", "w")
for (dirpath, subdirs, filenames) in os.walk(train_path):
    filenames = sorted(filenames)
    all_lines = []
    out_line = ""
    for file in filenames:
        img_ext = file.split(".")[-1]
        if img_ext == "png":
            #print(file)
            dir_p = dirpath.split("/")
            folder =  "/".join([dir_p[-4], dir_p[-3], dir_p[-2] , dir_p[-1]])
            frame_index = file.split(".")[0].split("_")[-1]
            out_line = " ".join([folder, frame_index, "1"])
            all_lines.append(out_line)
    all_lines = all_lines[1:-1]
    w_lines = map(lambda x:x+'\n', all_lines)
    train_file.writelines(w_lines)  
train_file.close()


      
