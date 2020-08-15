# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# To run distributed training on SuperServer:
# Note: There's some error in PoseDecoder during distributed training. Use posecnn instead
# CUDA_VISIBLE_DEVICES=0,1 python train.py --data_path ../../data/kitti-raw/ --png --use_fastdepth --distributed --pose_model_type posecnn --batch_size 6 --log_dir tmp/fastdepth

# python train.py --data_path ../../data/kitti-raw/ --png --depth_model_arch pydnet --pose_model_type posecnn --batch_size 8 --distributed --log_dir tmp/pydnet_amp --gpu 0 1 --amp

from __future__ import absolute_import, division, print_function

import os
from options import MonodepthOptions

import numpy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer import Trainer

import datasets
import networks

from utils import *
from datasets.kitti_utils import *
from networks.layers import *

from networks import models as fastdepth
from networks.pydnet import pydnet

from torch.utils.data import DataLoader


def init_process(rank, opt, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # create default process group
    dist.init_process_group(backend, rank=rank, world_size=opt.world_size)
    fn(rank, opt)

def setup(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    # create default process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def main_worker(rank, opt):

    if opt.distributed:
        setup(rank, opt.world_size, backend='nccl')
    opt.rank = rank
    num_input_frames = len(opt.frame_ids)
    num_pose_frames = 2 if opt.pose_model_input == "pairs" else num_input_frames

    models = {}
    if opt.depth_model_arch in ["fastdepth", "mobilenet"]:
        models["fastdepth"] = fastdepth.MobileNetSkipAddMultiScale(True,
                            pretrained_path = os.path.join("networks", 'imagenet', 
                 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256',
                 'model_best.pth.tar'), scales = opt.scales)
        # models["fastdepth"] = fastdepth.MobileNetSkipAddConvTMultiScale(True,
        #                     pretrained_path = os.path.join("networks", 'imagenet', 
        #          'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256',
        #          'model_best.pth.tar'), scales = opt.scales)
    elif  opt.depth_model_arch == "pydnet":
        models["fastdepth"] = pydnet.Pyddepth(opt.scales, mobile_version = False, my_version=False)
    else:
        models["encoder"] = networks.ResnetEncoder(opt.num_layers, True)

        models["depth"] = networks.DepthDecoder(models["encoder"].num_ch_enc,
                                                opt.scales) # Niantic

    opt.use_pose_net = not (opt.use_stereo and opt.frame_ids == [0])
    
    if opt.use_pose_net:
        if opt.pose_model_type == "separate_resnet":
            models["pose_encoder"] = networks.ResnetEncoder(
                opt.num_layers,
                opt.weights_init == "pretrained",
                num_input_images=num_pose_frames)

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
                models["encoder"].num_ch_enc, num_pose_frames)

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

    if opt.gpu: # is not None
        if opt.distributed:
            for k, v in models.items():
                #models[k].share_memory()
                models[k] = DDP(models[k].to(opt.rank), device_ids=[opt.rank], # 1 process per gpu
                                broadcast_buffers=False, find_unused_parameters=True) ## Multiple GPU)
                
            # opt.batch_size *= 2
            print("Using DistributedDataParallel with batch_size", opt.batch_size)
        else:
            if opt.world_size > 1:
                if opt.amp:
                    raise(NotImplementedError, "DataParallel does not work with AMP yet. Check https://pytorch.org/docs/stable/notes/amp_examples.html#dataparallel-in-a-single-process")
                for k, v in models.items():
                    models[k] = torch.nn.DataParallel(models[k]).cuda() #Data parallel
                opt.batch_size *= 2
                print("Using DataParallel with batch_size", opt.batch_size)
            else:
                for k, v in models.items():
                    models[k] = models[k].cuda()

    #Dataset
    fpath = os.path.join(os.path.dirname(__file__),"splits", opt.split, "{}_files.txt")

    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.png' if opt.png else '.jpg'

    num_train_samples = len(train_filenames)
    num_total_steps = num_train_samples // opt.batch_size * opt.num_epochs
    print(num_total_steps, "training steps")

    dataset = datasets.KITTIRAWDataset # Niantic
    train_dataset = dataset(
        opt.data_path, train_filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=True, img_ext=img_ext)

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    else:
        train_loader = DataLoader(
            train_dataset, opt.batch_size, True,
            num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_dataset = dataset(
        opt.data_path, val_filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=False, img_ext=img_ext)
    val_loader = DataLoader(
        val_dataset, opt.batch_size, True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    # print("rank", rank)
    # torch.autograd.set_detect_anomaly(True) # To detect if some errors in Distributed training

    trainer = Trainer(models, train_loader, val_loader, opt)
    trainer.train()
    
    if opt.distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":

    options = MonodepthOptions()
    opt = options.parse()

    if opt.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join(str(x) for x in opt.gpu) # "0,1"
        assert torch.cuda.is_available(), "No GPU detected. Remove --gpu argument and try again"

        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    opt.world_size = torch.cuda.device_count()

    if opt.distributed:
        
        # Processes
        # torch.multiprocessing.set_start_method('spawn')

        # processes = []

        # for rank in range(opt.world_size):
        #     p = Process(target=init_process, args=(rank, opt, main_worker))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()

        mp.spawn(main_worker,
                 args=(opt,),
                 nprocs=opt.world_size, 
                 join=True)
    else:
        main_worker(None, opt)
