from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

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
    
    
    def get_image_path(self, folder, frame_index, side, mode):
        f_str = folder + "_{:06d}_leftImg8bit.png".format(frame_index)
        image_path = os.path.join(self.data_path, "leftImg8bit_sequence", mode, f_str)
        
        return image_path
    
    
    def get_color(self, folder, frame_index, side, do_flip, mode):
        color = self.loader(self.get_image_path(folder, frame_index, side, mode))

        if do_flip and color is not None:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
        

