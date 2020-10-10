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


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip, mode):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        
        depth_gt = skimage.transform.resize(
            depth_gt, (self.height, self.width), order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps.
    Assumes the [gt depths dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) to be in the Kitti folder:
    After downloading the dataset, combine the train and val folder into gtdepths folder and place it in the kitti-raw folder
    
    Not all KITTI images have gtdepths. If not available, we load the velodyne data. 
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        subfolder=os.path.split(folder)[1]
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            'gtdepths',
            subfolder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        try:
            depth_gt = pil.open(depth_path)
            depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
            depth_gt = np.array(depth_gt).astype(np.float32) / 256
        except FileNotFoundError: # since not all gt data is available
            calib_path = os.path.join(self.data_path, folder.split("/")[0])

            velo_filename = os.path.join(
                self.data_path,
                folder,
                "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

            depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])

        depth_gt = skimage.transform.resize(
            depth_gt, (self.height, self.width), order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    
class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps.
    Assumes the [gt depths dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) to be in the Kitti folder.
    After downloading the dataset, combine the train and val folder into gtdepths folder and place it in the kitti-raw folder
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip, get_dense=False):
        subfolder=os.path.split(folder)[1]
        f_str = "{:010d}.png".format(frame_index)
        if get_dense:
            depth_path = os.path.join(
                        self.data_path,
                        'gtdepths',
                        subfolder,
                        "proj_depth/densified/image_0{}".format(self.side_map[side]),
                        f_str)
        else:
            depth_path = os.path.join(
                self.data_path,
                'gtdepths',
                subfolder,
                "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
                f_str)

        # import pdb
        # pdb.set_trace()
        try:
            depth_gt = pil.open(depth_path)
            # depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
            depth_gt = np.array(depth_gt).astype(np.float32) / 256
        except FileNotFoundError: # since not all gt data is available
            calib_path = os.path.join(self.data_path, folder.split("/")[0])

            velo_filename = os.path.join(
                self.data_path,
                folder,
                "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

            depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])

        depth_gt = skimage.transform.resize(
            depth_gt, (self.height, self.width), order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

class KITTIDenseDepthDataset(KITTIDepthDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIDenseDepthDataset, self).__init__(*args, **kwargs)

    def get_depth(self, folder, frame_index, side, do_flip):
        return super().get_depth(self, folder, frame_index, side, do_flip, get_dense=True)