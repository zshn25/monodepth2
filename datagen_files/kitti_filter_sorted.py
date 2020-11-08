import os

open('filter.txt','w').writelines([ line for line in open('splits/kitti_split_sorted/train_files.txt') if 'image_02' in line])
