# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# python train.py --data_path ../../data/kitti-raw/ --png --depth_model_arch pydnet --pose_model_type posecnn --batch_size 8 --distributed --log_dir tmp/pydnet_amp --gpu 0 1 --amp


from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *
from datasets.kitti_utils import *
from networks.layers import *

import datasets
import networks
from networks.fastdepth import models as fastdepth
from networks.pydnet.pydnet import Pyddepth

def setup(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # create default process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def main_worker(rank, opt):
    opt.rank = rank
    if opt.distributed:
        setup(opt.rank, opt.world_size, "nccl")

    opt.num_scales = len(opt.scales)
    num_input_frames = len(opt.frame_ids)
    opt.num_pose_frames = 2 if opt.pose_model_input == "pairs" else num_input_frames

    assert opt.frame_ids[0] == 0, "frame_ids must start with 0"

    opt.use_pose_net = not (opt.use_stereo and opt.frame_ids == [0])

    if opt.use_stereo:
        opt.frame_ids.append("s")

    # Models
    models = {}
    if opt.depth_model_arch == "fastdepth":
        models["fastdepth"] = fastdepth.MobileNetSkipAdd([], 
                                                                False, 
                                                                "",
                                                                opt.scales)
    elif opt.depth_model_arch == "pydnet":
        models["fastdepth"] = Pyddepth(opt.scales, 
                                mobile_version = True, 
                                my_version=False)
    else:
        models["encoder"] = networks.ResnetEncoder(
            opt.num_layers, opt.weights_init == "pretrained")
        
        models["depth"] = networks.DepthDecoder(
            models["encoder"].num_ch_enc, opt.scales)

    if opt.use_pose_net:
        if opt.pose_model_type == "separate_resnet":
            models["pose_encoder"] = networks.ResnetEncoder(
                opt.num_layers,
                opt.weights_init == "pretrained",
                num_input_images=opt.num_pose_frames)

            models["pose"] = networks.PoseDecoder(
                models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        elif opt.pose_model_type == "shared":
            if not hasattr(models, "encoder"):
                models["encoder"] = networks.ResnetEncoder(
                                        opt.num_layers,
                                        opt.weights_init == "pretrained")

            models["pose"] = networks.PoseDecoder(
                models["encoder"].num_ch_enc, opt.num_pose_frames)

        elif opt.pose_model_type == "posecnn":
            models["pose"] = networks.PoseCNN(
                num_input_frames if opt.pose_model_input == "all" else 2)

    if opt.predictive_mask:
        assert opt.disable_automasking, \
            "When using predictive_mask, please disable automasking with --disable_automasking"

        # Our implementation of the predictive masking baseline has the the same architecture
        # as our depth decoder. We predict a separate mask for each source frame.
        models["predictive_mask"] = networks.DepthDecoder(
            models["encoder"].num_ch_enc, opt.scales,
            num_output_channels=(len(opt.frame_ids) - 1))

    # Distributed
    if opt.gpu: # is not None
        if opt.distributed:
            for model_name in models.keys():
                models[model_name] = DDP(models[model_name].to(opt.rank), device_ids=[opt.rank])

    # Dataset
     # data
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                        "kitti_odom": datasets.KITTIOdomDataset}
    dataset = datasets_dict[opt.dataset]

    fpath = os.path.join(os.path.dirname(__file__), "splits", opt.split, "{}_files.txt")

    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.png' if opt.png else '.jpg'

    num_train_samples = len(train_filenames)
    opt.num_total_steps = num_train_samples // opt.batch_size * opt.num_epochs

    train_dataset = dataset(
        opt.data_path, train_filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=True, img_ext=img_ext)
    train_loader = DataLoader(
        train_dataset, opt.batch_size, True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_dataset = dataset(
        opt.data_path, val_filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=False, img_ext=img_ext)
    val_loader = DataLoader(
        val_dataset, opt.batch_size, True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    # torch.autograd.set_detect_anomaly(True)

    trainer = Trainer(opt, models, train_loader, val_loader)
    trainer.train()

    if opt.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":

    options = MonodepthOptions()
    opt = options.parse()

    if opt.gpu: #is not None
        opt.world_size = len(opt.gpu)
        
        os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join(str(x) for x in opt.gpu) # "0,1"
        assert torch.cuda.is_available(), "No GPU detected. Remove --gpu argument and try again"

        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if opt.distributed:
            mp.spawn(main_worker,
                    args=(opt,),
                    nprocs=1, # 1 process per gpu 
                    join=True)
        else:
            main_worker(None, opt)
    else:
        main_worker(None, opt)

   