from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class ZeroBikeDataset(MonoDataset):
    """Superclass for different types of Cityscapes dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(ZeroBikeDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[655.4564/1920, 0, 939.3567/1920, 0],
                           [0, 655.2325/1080, 536.3076/1080, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32) 

        self.full_res_shape = (1920, 1080)
    
    
    def get_image_path(self, folder, frame_index, side, mode):
        f_str = "frame_{:06d}.png".format(frame_index)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    
    def get_color(self, folder, frame_index, side, do_flip, mode):
        color = self.loader(self.get_image_path(folder, frame_index, side, mode))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
