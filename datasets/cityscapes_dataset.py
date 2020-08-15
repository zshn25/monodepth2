# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {None:None}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        assert(folder.split("/")[1] == 'cityscapes')
        f_str = "{}_{}_leftImg8bit.png".format(folder.split("/")[-1],frame_index)
        image_path = os.path.join(self.data_path,
            folder + '/', f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        assert(folder.split('/')[1] == 'cityscapes')

        calib_path = os.path.join(self.data_path, '/'.join(folder.split("/")[0:3]))
        
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{}.bin".format(frame_index))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
