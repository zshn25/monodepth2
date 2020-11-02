from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class YamahaDataset(MonoDataset):
    """Superclass for different types of Cityscapes dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(YamahaDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[1.104, 0, 0.5, 0],
                           [0, 2.212, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32) 

        self.full_res_shape = (1280, 720)
    
    
    def get_image_path(self, folder, frame_index, side, mode):
        f_str = folder + "_{:06d}.png".format(frame_index)
        image_path = os.path.join(self.data_path, f_str)
        
        return image_path
    
    
    def get_color(self, folder, frame_index, side, do_flip, mode):
        color = self.loader(self.get_image_path(folder, frame_index, side, mode))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
