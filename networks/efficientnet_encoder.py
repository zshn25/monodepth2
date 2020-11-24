# Copyright ^Zeeshan Khan Suri 2020

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn

import timm

class EfficientnetEncoder(nn.Module):
    """Pytorch module for a EfficientNet-Egde-Small encoder. Requires timm package
    Inputs:
        small_size:bool: if True, loads EffecientNet-Edge-Small else Medium
    """
    def __init__(self, small_version=False, pretrained=False):
        super(EfficientnetEncoder, self).__init__()

        #self.num_ch_enc = np.array([32, 32, 48, 144, 192]) # for "efficientnet_es" and "efficientnet_em"
        self.num_ch_enc = np.array([32, 24, 40, 112, 320])
        model = timm.create_model('efficientnet_lite0', pretrained=pretrained) # edge models -> "efficientnet_es" and "efficientnet_em" but slower on PC

        self.layer1 = nn.Sequential(model.conv_stem,model.bn1,model.act1)
        self.layer2 = model.blocks[:2]
        self.layer3 = model.blocks[2:3]
        self.layer4 = model.blocks[3:5]
        self.layer5 = model.blocks[5:]

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features.append(self.layer1(x))
        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))
        self.features.append(self.layer5(self.features[-1]))

        return self.features

