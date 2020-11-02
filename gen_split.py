# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:28:05 2020

@author: chand
"""

### Place it in videos folder

import os
import argparse
import platform
from glob import glob

file_dir = os.path.dirname(__file__)
splits_path = os.path.join(file_dir, "splits")

parser = argparse.ArgumentParser(description="Split options")  
  
## Choices
parser.add_argument("--choices", nargs="+", type=int, 
                    help="selected datasets for custom split file - \
                        datasets: {1: Cityscapes, 2: Yamaha dataset, 3: Zero dataset}", 
                    default=[3])


## Cityscapes dataset
parser.add_argument("--cityscapes_data_path", type=str,
                    help="path to the cityscapes training data",
                    default= "../datasets/cityscapes_data")

parser.add_argument("--cityscapes_split_path",type=str,
                    help="path to the cityscapes split file",
                    default=os.path.join(splits_path, "cityscapes_split"))

parser.add_argument("--gen_city_split", 
                    help="generate cityscapes dataset split",
                    action="store_true", default = False)

## Yamaha dataset
parser.add_argument("--yamaha_data_path", type=str,
                    help="path to the Yamaha training data",
                    default="../datasets/yamaha_dataset")

parser.add_argument("--yamaha_split_path", type=str,
                    help="path to the yamaha dataset split file",
                    default=os.path.join(splits_path, "yamaha_split"))

parser.add_argument("--gen_yamaha_split", 
                    help="generate yamaha dataset split",
                    action="store_true", default = False)

## Zero dataset
parser.add_argument("--zero_data_path", type=str,
                    help="path to the Zero training data",
                    default="./zero_data")

parser.add_argument("--zero_split_path", type=str,
                    help="path to the zero dataset split file",
                    default=os.path.join(splits_path, "zero_split"))

parser.add_argument("--gen_zero_split", 
                    help="generate zero dataset split",
                    action="store_true", default = True)


opts = parser.parse_args()



def gen_city_split_data():
    
    train_path = os.path.join(opts.cityscapes_data_path, "leftImg8bit_sequence", "train")
    val_path = os.path.join(opts.cityscapes_data_path, "leftImg8bit_sequence", "val")
       
    ## writing cityscapes train files
    t_path = opts.cityscapes_split_path + os.sep + "train_files.txt"
    train_file = open(t_path, "w")
    for (_, _, filenames) in os.walk(train_path):
        filenames = sorted(filenames)
        for file in filenames:
            folder = file.split("_")[0]
            img_head = "_".join(file.split("_", 2)[:2])
            img_head =  "/".join([folder, img_head])
            frame_index = file.split("_")[-2].lstrip("0") or "0"
            out_line = " ".join([img_head, frame_index, "1"])
            train_file.write(out_line + "\n")
            
    train_file.close()
    refine_city_data(t_path)
    
    ## writing cityscapes val files
    v_path = opts.cityscapes_split_path + os.sep + "val_files.txt"
    val_file = open(v_path, "w")
    for (_, _, filenames) in os.walk(val_path):
        filenames = sorted(filenames)
        for file in filenames:
            folder = file.split("_")[0]
            img_head = "_".join(file.split("_", 2)[:2])
            img_head =  "/".join([folder, img_head])
            frame_index = file.split("_")[-2].lstrip("0") or "0"
            out_line = " ".join([img_head, frame_index, "1"])
            val_file.write(out_line + "\n")
        
    val_file.close()
    refine_city_data(v_path)

def gen_yamaha_split_data():
    
    train_path = os.path.join(opts.yamaha_data_path, "train")
    val_path = os.path.join(opts.yamaha_data_path, "val")
    
    ## writing yamaha train files
    train_file = open(opts.yamaha_split_path + os.sep + "train_files.txt", "w")
    for (_, _, filenames) in os.walk(train_path):
        filenames = sorted(filenames)
        all_lines = []
        out_line = ""
        for file in filenames:
            folder = file.split("_")[0] 
            img_head =  "/".join([folder, folder])
            frame_index = file.split("_")[-1].split(".")[0].lstrip("0") or "0"
            out_line = " ".join([img_head, frame_index, "1"])
            all_lines.append(out_line)
        all_lines = all_lines[1:-1]
        w_lines = map(lambda x:x+'\n', all_lines)
        train_file.writelines(w_lines)  
    train_file.close()
    
    ## writing yamaha val files
    val_file = open(opts.yamaha_split_path + os.sep + "val_files.txt", "w")
    for (_, _, filenames) in os.walk(val_path):
        filenames = sorted(filenames)
        for file in filenames:
            folder = file.split("_")[0] 
            img_head =  "/".join([folder, folder])
            frame_index = file.split("_")[-1].split(".")[0].lstrip("0") or "0"
            out_line = " ".join([img_head, frame_index, "1"])
            all_lines.append(out_line)
        all_lines = all_lines[1:-1]
        w_lines = map(lambda x:x+'\n', all_lines)
        val_file.writelines(w_lines)  

    val_file.close()

def gen_zero_split_data():
    
    train_path = os.path.join(opts.yamaha_data_path, "train")
    val_path = os.path.join(opts.yamaha_data_path, "val")
    
    ## writing yamaha train files
    train_file = open(opts.yamaha_split_path + os.sep + "train_files.txt", "w")
    for (_, _, filenames) in os.walk(train_path):
        filenames = sorted(filenames)
        all_lines = []
        out_line = ""
        for file in filenames:
            img_head, frame_name = file.split("_")
            frame_index = frame_name.split('.')[0]
            out_line = " ".join([img_head, frame_index, "1"])
            all_lines.append(out_line)
        all_lines = all_lines[1:-1]
    
    data_size = len(all_lines)
    train_size = int(data_size * 0.8)
    train_lines = all_lines[:train_size]
    val_lines = all_lines[train_size:]
    
    train_w_lines = map(lambda x:x+'\n', train_lines)
    train_file.writelines(train_w_lines)  
    train_file.close()
    
    ## writing yamaha val files
    val_file = open(opts.yamaha_split_path + os.sep + "val_files.txt", "w")
    val_w_lines = map(lambda x:x+'\n', val_lines)
    val_file.writelines(val_w_lines)  

    val_file.close()


def refine_city_data(path):
    file = open(path, "r")
    lines = file.readlines()
    all_list = []
    folder_list = []
    for line in lines:
        req = line.split(" ")[0].split("/")[1]
        if not req in folder_list:
            if all_list is not None:
                all_list = all_list[:-1]
            folder_list.append(req)
        else:
            all_list.append(line)
   
    file.close()
    file = open(path, "w")
    file.writelines(all_list[:-1])
    file.close()
    

    
def main():
    choices = opts.choices 
    for choice in choices:
        if choice == 1:
            if opts.gen_city_split:
                gen_city_split_data()

        if choice == 2:
            if opts.gen_yamaha_split:
                gen_yamaha_split_data()
                
        if choice == 3:
            if opts.gen_zero_split:
                gen_zero_split_data()
    
        
if __name__ == "__main__":
    main()