from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class ZeroDataset(MonoDataset):
    """Superclass for different types of Cityscapes dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(ZeroDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[655.4564180577918, 0.0, 939.3566840070027, 0], 
                           [0.0, 655.232493150707, 536.3076255862055, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)  

        # Oriignal Image
        self.K[0,0] /= 1920
        self.K[0,2] /= 1920
        self.K[1,1] /= 1080 
        self.K[1,2] /= 1080 

        self.full_res_shape = (1920, 1080)
    
    
    def get_image_path(self, folder, frame_index, side, mode):
        f_str = folder + "_{}.jpg".format(frame_index)
        image_path = os.path.join(self.data_path, 'train', f_str)
        
        return image_path
    
    
    def get_color(self, folder, frame_index, side, do_flip, mode):
        color = self.loader(self.get_image_path(folder, frame_index, side, mode))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

