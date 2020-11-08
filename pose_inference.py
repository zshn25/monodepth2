# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:25:48 2020

@author: chand
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', default = "img/")
    parser.add_argument('--model_path', type=str,
                        help='name of a pretrained model to use', default = "weights")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument('--feed_height', type=int,
                        help='monodepth input height', default=192)
    parser.add_argument('--feed_width', type=int,
                        help='monodepth input width', default=640)
    
    return parser.parse_args()

def pose_inference(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter; see README.md for an example"
        
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    
    model_path = os.path.join(args.model_path, "models")
    print("-> Loading model from ", model_path)
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")
    
    # LOADING PRETRAINED MODEL
    
    pose_encoder = networks.ResnetEncoder(18, False, 2)
    loaded_dict_enc = torch.load(pose_encoder_path, map_location=device)
    #filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in pose_encoder.state_dict()}
    pose_encoder.load_state_dict(loaded_dict_enc)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    
    print("Loading pose decoder network")
    pose_decoder = networks.PoseDecoderOld(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    loaded_dict = torch.load(pose_decoder_path, map_location=device)
    pose_decoder.load_state_dict(loaded_dict)
    
    pose_decoder.to(device)
    pose_decoder.eval()
    
    
        # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))
    
    
     # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx in range(len(paths)-1):
            
            # Load image and preprocess
            input_image_1 = pil.open(paths[idx]).convert('RGB')
            input_image_2 = pil.open(paths[idx+1]).convert('RGB')
            assert input_image_1.size==input_image_2.size,\
            "The dimensions of both input images must be the same"
            
            input_image_1 = input_image_1.resize((args.feed_width, args.feed_height), pil.LANCZOS)
            input_image_1 = transforms.ToTensor()(input_image_1).unsqueeze(0)
            
            input_image_2 = input_image_2.resize((args.feed_width, args.feed_height), pil.LANCZOS)
            input_image_2 = transforms.ToTensor()(input_image_2).unsqueeze(0)
            
            input_frames = torch.cat([input_image_1,input_image_2],1)
            input_frames = input_frames.to(device)
                      
            # PREDICTION
            features = pose_encoder(input_frames)
            axisangle,translation = pose_decoder([features])
            
            print("Axisangle and translation for frame_{} and frame_{} is: {},{}".format(idx, idx+1, axisangle, translation))
            
            
            
if __name__ == '__main__':
    args = parse_args()
    pose_inference(args)
