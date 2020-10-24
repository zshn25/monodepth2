# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

from time import time

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

# Segmentation classes for cityscapes dataset
class_names = [
                        "unlabelled",
                        "road",
                        "sidewalk",
                        "building",
                        "wall",
                        "fence",
                        "pole",
                        "traffic_light",
                        "traffic_sign",
                        "vegetation",
                        "terrain",
                        "sky",
                        "person",
                        "rider",
                        "car",
                        "truck",
                        "bus",
                        "train",
                        "motorcycle",
                        "bicycle"
                  ]
segment_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
colors = [  #[  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]
        ]
label_colours = dict(zip(range(19), colors))

def decode_mask(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(len(segment_classes)):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def get_bboxes_seg(mask_outputs, im):
    '''OpenCV single class multi-instance per image approach'''
    bboxes = []            
    ins_mask_pred = np.squeeze(mask_outputs[("ins_mask", 0)].data.max(1)[1].cpu().numpy() + 1, 0)
    seg_mask_pred = mask_outputs[("seg_mask", 0)].data.squeeze()
    max_seg_mask_pred = seg_mask_pred.max(0)[0].unsqueeze(0)
    
    seg_mask_pred = (seg_mask_pred == max_seg_mask_pred).cpu().numpy().astype(int)
    for idx in range(11, seg_mask_pred.shape[0]):
        seg_mask = seg_mask_pred[idx] * ins_mask_pred
        
        print(np.unique(seg_mask), np.amin(seg_mask), np.amax(ins_mask_pred), np.amax(seg_mask_pred[idx]),  seg_mask.shape)
        seg_mask = seg_mask.astype(np.uint8)
        
        contours, heirarchy = cv2.findContours(seg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("Show",seg_mask)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),2)
    cv2.imshow("Show",im)
    cv2.waitKey(0)
    return bboxes
    
def get_bboxes(mask_outputs, im, output_name, output_directory):
    '''OpenCV single instance multi-class per image approach (default)'''
    start = time()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) 
    bboxes = []        
    
    seg_mask_pred = np.squeeze(mask_outputs[("seg_mask", 0)].data.max(1)[1].cpu().numpy(), 0)
    ins_mask_pred = mask_outputs[("ins_mask", 0)].data.squeeze()
    max_ins_mask_pred, max_ins_mask_pred_idx = ins_mask_pred.max(0)
    
    # max_ins_mask_pred[max_ins_mask_pred_idx < class_names.index('person')] = -9999

    seg_mask_pred[seg_mask_pred < class_names.index('sky')] = 0
    
    start = time()    
    ins_mask_pred = (ins_mask_pred == max_ins_mask_pred[None]).cpu().numpy().astype(int)
    
    for idx in range(ins_mask_pred.shape[0]):
        #if np.sum(ins_mask_pred[idx]) < 100:
        #    continue
        ins_mask = ins_mask_pred[idx] * seg_mask_pred
        
        ins_mask = ins_mask.astype(np.uint8)
        # print(np.amax(ins_mask), np.amin(ins_mask), np.unique(ins_mask))

        contours, heirarchy = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #print(time() - start)  
        #start = time()
        for cnt in contours:
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 and h < 5:
                continue
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bboxes.append([x, y, w, h])
    print(time() - start)  
    # print(bboxes)
    # print(time() - start)  
              
    # cv2.imshow("Show", im)
    save_path = os.path.join(output_directory, "{}_ins_mask_bbox.jpeg".format(output_name))
    cv2.imwrite(save_path, im)
    return bboxes


def f(ins_mask):
   contours, heirarchy = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   return contours

p = Pool(4)
def get_bboxes_threads(mask_outputs, im, output_name, output_directory):
    '''OpenCV single instance single-class per image multi-threaded approach'''
    start = time()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) 
    bboxes = []            
    seg_mask_pred = np.squeeze(mask_outputs[("seg_mask", 0)].data.max(1)[1].cpu().numpy(), 0)
    ins_mask_pred = mask_outputs[("ins_mask", 0)].data.squeeze()
    max_ins_mask_pred, max_ins_mask_pred_idx = ins_mask_pred.max(0)
    
    # max_ins_mask_pred[max_ins_mask_pred_idx < class_names.index('person')] = -9999

    seg_mask_pred[seg_mask_pred < class_names.index('sky')] = 0
    ins_mask_pred = (ins_mask_pred == max_ins_mask_pred[None]).cpu().numpy().astype(int)
    
 
    ins_mask_arr = (ins_mask_pred * seg_mask_pred[None]).astype(np.uint8)
    contours_list = p.map(f, ins_mask_arr)
    
    print(time() - start)  
    start = time()
    for contours in contours_list:
        for cnt in contours:
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 and h < 5:
                continue
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bboxes.append([x, y, w, h])
    print(time() - start)  
    print(bboxes)
              
    # cv2.imshow("Show", im)
    save_path = os.path.join(output_directory, "{}_ins_mask_bbox.jpeg".format(output_name))
    cv2.imwrite(save_path, im)
    return bboxes


def bbox2(img):
    '''Numpy approach'''
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)
    
def get_bboxes1(mask_outputs, im, output_name, output_directory):
    '''Numpy approach'''
    start = time()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) 
    bboxes = []            
    seg_mask_pred = mask_outputs[("seg_mask", 0)].data.squeeze()
    max_seg_mask_pred, max_seg_mask_pred_idx = seg_mask_pred.max(0)
    seg_mask_pred = (seg_mask_pred == max_seg_mask_pred[None]).cpu().numpy().astype(int)
    
    # seg_mask_pred[max_seg_mask_pred_idx <= class_names.index('sky') + 1] = 0
    
    ins_mask_pred = mask_outputs[("ins_mask", 0)].data.squeeze()
    max_ins_mask_pred, max_ins_mask_pred_idx = ins_mask_pred.max(0)
    ins_mask_pred = (ins_mask_pred == max_ins_mask_pred[None]).cpu().numpy().astype(int)
    
    seg_start_idx = class_names.index('person') - 1
    for ins_idx in range(ins_mask_pred.shape[0]):
        for seg_idx in range(seg_start_idx, seg_mask_pred.shape[0]):
            
            img = ins_mask_pred[ins_idx] * seg_mask_pred[seg_idx]
            #cv2.imwrite('./save_img/{}_{}.jpg'.format(ins_idx, seg_idx), img * 255)    
            if np.amax(img) == np.amin(img):
                continue
            
            start = time()
            x, y, w, h = bbox2(img)
            print(time() - start)  
            
            if w < 15 and h < 5:
                continue
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bboxes.append([x, y, w, h])
            
    print(bboxes)
              
    # cv2.imshow("Show", im)
    save_path = os.path.join(output_directory, "{}_ins_mask_bbox.jpeg".format(output_name))
    cv2.imwrite(save_path, im)
    return bboxes

    
def get_bboxes_new(mask_outputs, im, output_name, output_directory):
    start = time()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) 
    bboxes = []     
          
    seg_mask_pred = mask_outputs[("seg_mask", 0)].data.squeeze()
    max_seg_mask_pred, max_seg_mask_pred_idx = seg_mask_pred.max(0)
    
    # seg_mask_pred[max_seg_mask_pred_idx <= class_names.index('sky') + 1] = 0
    
    ins_mask_pred = mask_outputs[("ins_mask", 0)].data.squeeze()
    max_ins_mask_pred, max_ins_mask_pred_idx = ins_mask_pred.max(0)
      
    seg_start_idx = class_names.index('person') - 1
    max_seg_mask_pred_idx[max_seg_mask_pred_idx < seg_start_idx] = 0
    ins_mask = (max_seg_mask_pred_idx * max_seg_mask_pred_idx * (max_ins_mask_pred_idx + 1)) % 256
    ins_mask = ins_mask.cpu().numpy().astype(np.uint8)

    start = time()   
    contours, heirarchy = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        #if w < 15 and h < 5:
        #    continue
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bboxes.append([x, y, w, h])
    print(time() - start)  
        
    print(bboxes)
              
    # cv2.imshow("Show", im)
    save_path = os.path.join(output_directory, "{}_ins_mask_bbox.jpeg".format(output_name))
    cv2.imwrite(save_path, im)
    return bboxes

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "recent",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    mask_decoder_path = os.path.join(model_path, "mask.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained depth decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    print("   Loading pretrained mask decoder")
    mask_decoder = networks.MaskDecoder(num_ch_enc = encoder.num_ch_enc, scales = range(4), num_output_channels = len(class_names))
    loaded_dict = torch.load(mask_decoder_path, map_location=device)
    mask_decoder.load_state_dict(loaded_dict)

    mask_decoder.to(device)
    mask_decoder.eval()

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
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
           
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            
            for _ in range(3):
                t1 = time()
                features = encoder(input_image)
                t2 = time()
                outputs = depth_decoder(features)
                t3 = time()
                mask_outputs = mask_decoder(features)
                t4 = time()
                print('Encoder time: {:.4f} Depth Decoder time: {:.4f} Mask Decoder time: {:.4f}'.format(t2 - t1, t3 - t2, t4 - t3))

            # Depth prediction
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)
            
            # Segmentation mask prediction
            seg_mask_pred = mask_outputs[("seg_mask", 0)]
            seg_mask_pred = np.squeeze(seg_mask_pred.data.max(1)[1].cpu().numpy(), axis = 0)

            seg_mask = decode_mask(seg_mask_pred) * 255 #* ins_mask_pred[..., None]
            seg_mask_im = pil.fromarray(seg_mask.astype(np.uint8))
            name_dest_seg_mask = os.path.join(output_directory, "{}_seg_mask.jpeg".format(output_name))
            seg_mask_im.save(name_dest_seg_mask)
            
                        
            # Instance mask prediction
            ins_mask_pred = mask_outputs[("ins_mask", 0)].data.max(1)[1].cpu().numpy()
            ins_mask_pred = (ins_mask_pred + 1) * seg_mask_pred[None]
            ins_mask_pred = np.squeeze(ins_mask_pred, axis = 0)
            
            # print(np.amax(seg_mask_pred), -np.amax(-seg_mask_pred))
            # print(np.amax(ins_mask_pred), -np.amax(-ins_mask_pred))
            
            objects = np.unique(ins_mask_pred.flatten())
            n_objects = len(objects) - 1 
            
            colors = [cm.Spectral(each) for each in np.linspace(0, 1, n_objects)]
            ins_mask_pred_color = np.zeros((ins_mask_pred.shape[0], ins_mask_pred.shape[1], 3), dtype=np.uint8)
            for i, obj in zip(range(n_objects), objects):
                ins_mask_pred_color[ins_mask_pred == obj] = (np.array(colors[i][:3]) * 255).astype('int')
                
            #ins_mask_pred = 0.3 + 0.7 * (ins_mask_pred / np.amax(ins_mask_pred)) 
            #ins_mask = ins_mask_pred * 255
            ins_mask_pred_color = ins_mask_pred_color.astype(np.uint8)
            for _ in range(3):
                t5 = time()
                get_bboxes(mask_outputs, ins_mask_pred_color, output_name, output_directory)
                t6 = time()
                print('Post processing time =  {:.4f}'.format(t6 - t5))
            
            ins_mask_im = pil.fromarray(ins_mask_pred_color)
            name_dest_ins_mask = os.path.join(output_directory, "{}_ins_mask.jpeg".format(output_name))
            ins_mask_im.save(name_dest_ins_mask)
            

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
