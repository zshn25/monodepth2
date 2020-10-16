from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class E2RDataset(MonoDataset):
    """Superclass for different types of Cityscapes dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(E2RDataset, self).__init__(*args, **kwargs)

        # For the stereo dataset
        self.K = np.array([[554.9437/1920, 0, 0.5, 0],
                           [0, 555.7158/1080, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32) 

        self.full_res_shape = (1920, 1080)
        self.side_map = {"l": "l", "r": "r"}
    
    
    def get_image_path(self, folder, frame_index, side, mode):
        #print(folder)
        sub_d = folder.split("/")[-1] + "_{}".format(side)
        #print(sub_d)
        f_str = folder + os.sep + sub_d  + os.sep + sub_d + "_{:06d}".format(frame_index) + self.img_ext
        image_path = os.path.join(self.data_path, mode, f_str)
        #print(image_path)
        return image_path
    
    
    def get_color(self, folder, frame_index, side, do_flip, mode):
        color = self.loader(self.get_image_path(folder, frame_index, side, mode))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

