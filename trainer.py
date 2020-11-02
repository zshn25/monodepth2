# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
if torch.__version__ < "1.6":
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

import json

import datasets
import networks

from utils import *
from networks.layers import *
from datasets.kitti_utils import *
from discriminative import DiscriminativeLoss

from IPython import embed

#from networks.fastdepth import models as fastdepth
#from networks.pydnet.pydnet import Pyddepth

from collections import OrderedDict

#import sys
#sys.path.append("../fastdepth") # Since cannot import from paths with '-'
#import models as fastdepth

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12351'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, rank, options):
        self.opt = options
        self.opt.rank = rank

        self.use_segmentation = self.opt.use_segmentation
        self.max_instances = self.opt.max_instances # Change also in data-generator
        self.num_classes = self.opt.num_classes

        # DDP setup
        if self.opt.distributed:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = "29501"
            dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=self.opt.world_size, rank=self.opt.rank)
            torch.cuda.set_device(self.opt.rank)
                
        self.device = torch.device("cpu" if self.opt.gpu is None else "cuda:{}".format(self.opt.rank) if self.opt.rank is not None else "cuda")

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Automatic mixed precision
        if self.opt.amp:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
                print("Using Automatic Mixed Precision")
            except Exception as e:
                raise(e, "Pytorhc version > 1.6 required for AMP")

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.depth_model_arch == "pydnet":
            self.models["fastdepth"] = Pyddepth(self.opt.scales, True, False)
            self.models["fastdepth"].to(self.device)
            if self.opt.distributed:
                self.models["fastdepth"] = DDP(self.models["fastdepth"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["fastdepth"].parameters())
        elif self.opt.depth_model_arch == "fastdepth":
            self.models["fastdepth"] = fastdepth.MobileNetSkipAddMultiScale(False,
                            pretrained_path = "", scales = self.opt.scales)

            self.models["fastdepth"].to(self.device)
            if self.opt.distributed:
                self.models["fastdepth"] = DDP(self.models["fastdepth"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["fastdepth"].parameters())

        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            if self.opt.distributed:
                self.models["encoder"] = DDP(self.models["encoder"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["encoder"].parameters())

            depth_num_ch_enc = self.models["encoder"].module.num_ch_enc if self.opt.distributed else self.models["encoder"].num_ch_enc
            
            self.models["depth"] = networks.DepthDecoder(depth_num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            if self.opt.distributed:
                self.models["depth"] = DDP(self.models["depth"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

                self.models["pose_encoder"].to(self.device)
                if self.opt.distributed:
                    self.models["pose_encoder"] = DDP(self.models["pose_encoder"],
                                                device_ids=[self.opt.rank], 
                                                broadcast_buffers=False, 
                                                find_unused_parameters=True) ## Multiple GPU
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            elif self.opt.pose_model_type == "shared":
                if not hasattr(self.models, "encoder"):
                    self.models["encoder"] = networks.ResnetEncoder(
                                            self.opt.num_layers,
                                            self.opt.weights_init == "pretrained")
                    self.models["encoder"].to(self.device)
                    if self.opt.distributed:
                        self.models["encoder"] = DDP(self.models["encoder"],
                                                    device_ids=[self.opt.rank], 
                                                    broadcast_buffers=False, 
                                                    find_unused_parameters=True) ## Multiple GPU
                    self.parameters_to_train += list(self.models["encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(depth_num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                assert not self.opt.train_intrinsics,\
                "Intrinsics network not compatible with PoseCNN"
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            if self.opt.distributed:
                self.models["pose"] = DDP(self.models["pose"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["pose"].parameters())
            
            if self.opt.train_intrinsics:
                self.resize_len = torch.tensor([[self.opt.width, self.opt.height]],device=self.device)
                self.models["intrinsics"] = networks.IntrinsicsNetwork(
                    self.models["encoder"].num_ch_enc,
                    self.resize_len)
                self.models["intrinsics"].to(self.device)
                if self.opt.distributed:
                    self.models["intrinsics"] = DDP(self.models["intrinsics"],
                                                    device_ids=[self.opt.rank], 
                                                    broadcast_buffers=False, 
                                                    find_unused_parameters=True) 
                self.parameters_to_train += list(self.models["intrinsics"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(depth_num_ch_enc, self.opt.scales,num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            if self.opt.distributed:
                self.models["predictive_mask"] = DDP(self.models["predictive_mask"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
            
        if self.use_segmentation:
            self.models["mask"] = networks.MaskDecoder(depth_num_ch_enc, self.opt.scales, num_output_channels=self.num_classes, n_objects = self.max_instances)
        
            self.models["mask"].to(self.device)
            if self.opt.distributed:
                self.models["mask"] = DDP(self.models["mask"],
                                               device_ids=[self.opt.rank], 
                                               broadcast_buffers=False, 
                                               find_unused_parameters=True) ## Multiple GPU
            self.parameters_to_train += list(self.models["mask"].parameters())
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)


        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        if self.opt.distributed:
            self.opt.batch_size = int(self.opt.batch_size / self.opt.world_size)
            self.opt.num_workers = int(self.opt.num_workers / self.opt.world_size)
            
        img_ext = '.png' if self.opt.png else '.jpg'

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "yamaha": datasets.YamahaDataset,
                         "zero": datasets.ZeroDataset,
                         "cityscapes": datasets.CityscapesDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        
        choices = OrderedDict([(0, "kitti"), (1, "cityscapes"), (2, "yamaha"), (3, "zero")])
        data_paths = {"kitti": self.opt.data_path,
                      "cityscapes": self.opt.cityscapes_data_path,
                      "yamaha": self.opt.yamaha_data_path,
                      "zero": self.opt.zero_data_path,}
        
        all_train_dataset = []
        all_val_dataset = []
        for choice in self.opt.choices:
            name = choices[choice]
            setattr(self, name, choice)
            if choice == 0:
                spath = os.path.join("splits", "{}_split", self.opt.split)
            else:
                spath = os.path.join("splits", "{}_split")
                
            if self.use_segmentation:
                f_format = "{}_files_seg.txt"
            else:
                f_format = "{}_files.txt"
                
            fpath = os.path.join(os.path.dirname(__file__), spath, f_format)
            train_filenames = readlines(fpath.format(name, "train"))
            val_filenames = readlines(fpath.format(name, "val"))
            
            dataset = datasets_dict[name]
            
            train_dataset = dataset(
            data_paths[name], train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, mode="train", 
            use_segmentation = self.use_segmentation, num_classes = self.num_classes, max_instances = self.max_instances)
            all_train_dataset = torch.utils.data.ConcatDataset([all_train_dataset, train_dataset])
                        
            val_dataset = dataset(
            data_paths[name], val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, mode="val", 
            use_segmentation = self.use_segmentation, num_classes = self.num_classes, max_instances = self.max_instances)
            all_val_dataset = torch.utils.data.ConcatDataset([all_val_dataset, val_dataset])
        
        num_train_samples = len(all_train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        self.train_loader = DataLoader(
            all_train_dataset,  self.opt.batch_size, shuffle=(train_sampler is None),
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

        self.val_loader = DataLoader(
            all_val_dataset, self.opt.batch_size, shuffle=(train_sampler is None),
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(all_train_dataset), len(all_val_dataset)))

        self.save_opts()


        self.criterion_discriminative = DiscriminativeLoss(delta_var = 0.5, delta_dist = 1.5, norm = 2, usegpu = True)
        self.train()

        if self.opt.distributed:
            dist.destroy_process_group()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                if self.opt.distributed:
                    if self.opt.rank == 0:
                        # All processes should see same parameters as they all start from same
                        # random parameters and gradients are synchronized in backward passes.
                        # Therefore, saving it in one process is sufficient.
                        self.save_model()
                else:
                    self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_optimizer.step()
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.opt.amp: # Automatic mixed precision
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.step(self.model_optimizer)
                self.scaler.update()
            else:
                losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                losses_list = [losses["loss"].cpu().data]
                if self.use_segmentation:
                    losses_list.extend([losses["seg_loss"].cpu().data, losses["ins_loss"].cpu().data])

                self.log_time(batch_idx, duration, losses_list)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        features = {}

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
            if self.use_segmentation:
                outputs.update(self.models["mask"](features[0]))
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            if self.opt.depth_model_arch in ["fastdepth", "pydnet"]:
                outputs = self.models["fastdepth"](inputs["color_aug", 0, 0])
            else:
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["depth"](features)
                if self.use_segmentation:
                    outputs.update(self.models["mask"](features))

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features={}):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if self.opt.amp:
                    with torch.cuda.amp.autocast(enabled=False):
                        outputs[("color", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            outputs[("sample", frame_id, scale)],
                            padding_mode="border", align_corners=True)
                        # inputs[("color", frame_id, source_scale)],
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales        
        
        if self.use_segmentation:
            seg_cce_loss = self.compute_cross_entropy(outputs[("seg_mask", 0)], inputs[("seg_mask", 0)])
            ins_cce_loss = self.criterion_discriminative(outputs[("ins_mask", 0)], inputs[("ins_mask", 0)], inputs["n_objects"], self.max_instances)
            losses["seg_loss"] = seg_cce_loss
            losses["ins_loss"] = ins_cce_loss
            total_loss += seg_cce_loss + ins_cce_loss
        
        losses["loss"] = total_loss
        losses["loss"] = total_loss
        return losses

    def compute_cross_entropy(self, pred, target, weight = None, size_average = True):
        n, c, h, w = pred.size()
        nt, ht, wt = target.size()

        if h != ht and w != wt:  
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(pred, target, weight = weight, size_average = size_average, ignore_index=250)
        
        return loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        # depth_pred = torch.clamp(F.interpolate(
        #     depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        if self.use_segmentation:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss total: {:.5f} seg:{:.5f} ins:{:.5f} | time elapsed: {} | time left: {}"
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, *loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
