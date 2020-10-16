from __future__ import absolute_import, division, print_function

import os
import sys
import torch
import skimage.transform
import numpy as np
import PIL.Image as pil
import matplotlib.cm as cm

from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class CityscapesDataset(MonoDataset):
    """Superclass for different types of Cityscapes dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[1.104, 0, 0.5, 0],
                           [0, 2.212, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32) 

        self.full_res_shape = (2048, 1024)
    
        
        self.ignore_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.segment_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
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

        self.class_map = dict(zip(self.segment_classes, range(len(self.segment_classes))))
        self.ignore_idx = 250
        colors = [  # [  0,   0,   0],
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

        self.label_colours = dict(zip(range(19), colors))
        
    
    def check_depth(self):
        return False
    
    def get_image_path(self, folder, frame_index, side):
    
        f_str = folder + "_{:06d}_leftImg8bit.png".format(frame_index)
        image_path = os.path.join(self.data_path, "leftImg8bit_sequence", self.mode, f_str)
        
        return image_path
    
    
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
        

    def decode(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(len(self.segment_classes)):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


    def encode_seg_mask(self, seg_mask):
        for cls in self.ignore_classes:
            seg_mask[seg_mask == cls] = self.ignore_idx
        for cls in self.segment_classes:
            seg_mask[seg_mask == cls] = self.class_map[cls]
        return seg_mask  
    
    def get_seg_mask_path(self, folder, frame_index, side, mode):
        f_str = folder + "_{:06d}_gtFine_labelIds.png".format(frame_index)
        seg_mask_path = os.path.join(self.data_path, "gtFine", mode, f_str)
        
        # img_path = self.get_image_path(folder, frame_index, side, mode)
        # seg_mask_path = re.sub('leftImg8bit_sequence', 'gtFine', img_path)
        # seg_mask_path = re.sub('leftImg8bit', 'gtFine', img_path)
        
        return seg_mask_path
    
    def get_seg_mask(self, seg_mask_path, do_flip, rsize = None):
        seg_mask = pil.open(seg_mask_path)

        classes = np.unique(np.array(seg_mask)) 
        if rsize is not None:
            seg_mask = seg_mask.resize(rsize, resample = pil.NEAREST)
            
        if not np.all(classes == np.unique(np.array(seg_mask))):
            # print("WARN: resizing yielded fewer classes")
            pass
            
        if do_flip:
            seg_mask = seg_mask.transpose(pil.FLIP_LEFT_RIGHT)

        return seg_mask
    
    def encode_ins_mask(self, ins_mask, seg_mask, path = None):
        instances = np.unique(ins_mask)
        for cls in self.ignore_classes:
            ins_mask[seg_mask == cls] = 0 # self.ignore_idx
        
        ins_mask[seg_mask == self.ignore_idx] = 0
            
        n_objects = 1
        for cls in self.segment_classes:  
            instances = np.unique(ins_mask[seg_mask == cls]).tolist()
            if len(instances) == 0:
                continue
            # index = np.argwhere(instances == self.ignore_idx)
            
            # instances.remove(self.ignore_idx)
            n_instances = len(instances)
            
            n_mapped = min(n_instances, self.max_instances)                                              
            class_map = dict(zip(sorted(instances), range(1, n_mapped)))
            
            # if len(instances) > self.n_objects:
            #    print(instances, path, len(instances))
            
            for num in instances:
                ins_mask[ins_mask == num] = class_map.get(num, 0)
                
            n_objects = max(n_objects, n_mapped)

        return ins_mask, n_objects
    
    def get_ins_path(self, folder, frame_index, side, mode):
        f_str = folder + "_{:06d}_gtFine_instanceIds.png".format(frame_index)
        seg_mask_path = os.path.join(self.data_path, "gtFine", mode, f_str)
        
        # img_path = self.get_image_path(folder, frame_index, side, mode)
        # seg_mask_path = re.sub('leftImg8bit_sequence', 'gtFine', img_path)
        # seg_mask_path = re.sub('leftImg8bit', 'gtFine', img_path)
        
        return seg_mask_path    

    def visualize(self, mask):
        ins_mask_pred = mask.numpy()
        n_objects = len(np.unique(ins_mask_pred.flatten())) - 1 
        colors = [cm.Spectral(each) for each in np.linspace(0, 1, n_objects)]
        ins_mask_pred_color = np.zeros((ins_mask_pred.shape[0], ins_mask_pred.shape[1], 3), dtype=np.uint8)
        for i in range(n_objects):
            ins_mask_pred_color[ins_mask_pred == (i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')
            
        ins_mask_im = pil.fromarray(ins_mask_pred_color.astype(np.uint8))
        name_dest_ins_mask = os.path.join('.', "{}_ins_mask.jpeg".format('sample'))
        ins_mask_im.save(name_dest_ins_mask)
        
        import sys
        sys.exit(0)

    def get_ins_mask(self, ins_mask_path, do_flip, rsize = None):
        ins_mask = pil.open(ins_mask_path)

        classes = np.unique(np.array(ins_mask)) 
        if rsize is not None:
            ins_mask = ins_mask.resize(rsize, resample = pil.NEAREST)
            
        if not np.all(classes == np.unique(np.array(ins_mask))):
            # print("WARN: resizing yielded fewer classes")
            pass
            
        if do_flip:
            ins_mask = ins_mask.transpose(pil.FLIP_LEFT_RIGHT)
             
        return ins_mask
         
    def get_masks(self, folder, frame_index, side, do_flip, mode, rsize = None):
        seg_mask_path = self.get_seg_mask_path(folder, frame_index, side, mode)
        if os.path.exists(seg_mask_path):
            seg_mask = self.get_seg_mask(seg_mask_path, do_flip, rsize)
            seg_mask = self.encode_seg_mask(np.array(seg_mask))
        else:
            seg_mask = np.ones((rsize[1], rsize[0])) * self.ignore_idx
            
        ins_mask_path = self.get_ins_path(folder, frame_index, side, mode)
        
        if os.path.exists(ins_mask_path):
            ins_mask = self.get_ins_mask(ins_mask_path, do_flip, rsize)
            ins_mask, n_objects = self.encode_ins_mask(np.array(ins_mask), seg_mask, ins_mask_path)
        else:
            ins_mask = np.ones((rsize[1], rsize[0])) * 0
            n_objects = 1
        
        # print(seg_mask_path, np.unique(seg_mask), np.unique(ins_mask))
        seg_mask = torch.from_numpy(seg_mask).long()
        ins_mask = torch.from_numpy(ins_mask).long()
        n_objects = torch.tensor(n_objects).long()
          
        
        #self.visualize(ins_mask)
        
        
        return seg_mask, ins_mask, n_objects
        
