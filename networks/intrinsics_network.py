# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:17:08 2020

@author: chand
"""


from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class IntrinsicsNetwork(nn.Module):
    def __init__(self, num_ch_enc, resize_len, stride=1):
        super(IntrinsicsNetwork, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.resize_len = resize_len
        
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("focal_lengths")] = nn.Conv2d(256, 2, 3, stride, 1)
        self.convs[("offsets")] = nn.Conv2d(256, 2, 3, stride, 1)
                
        self.network = nn.ModuleList(list(self.convs.values()))
        
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()


    def forward(self, bottleneck):
        
        self.intrinsics = {}
        
        ## intially obtain the shape of cat as [B, N, 1, 1] where
        ## B is batch size and N is number of channels
          
        last_features = [f[-1] for f in bottleneck]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
    
        cat = torch.mean(torch.mean(cat_features, 3), 2)
        cat = torch.unsqueeze(torch.unsqueeze(cat, 2), 3)
        x = cat
        
        foc_len = self.convs[("focal_lengths")](x)
        foc_len = self.softplus(foc_len)
        foc_len = torch.squeeze(torch.squeeze(foc_len,3),2)
        foc_len = foc_len * self.resize_len
        foci = torch.diag_embed(foc_len)

        offset = self.convs[("offsets")](x)
        offset = torch.squeeze(torch.squeeze((offset),3),2) + 0.5
        offset = offset * self.resize_len
        offset = torch.unsqueeze(offset, 2)

        intrinsics_mat = torch.cat([foci, offset],2)
        last_row = torch.tensor([[0.0, 0.0, 1.0]], device ="cuda")
        intrinsics_mat = torch.cat([intrinsics_mat, last_row.repeat(cat.shape[0], 1, 1)], 1)
        
        zero_t = torch.zeros(3,1,device ="cuda")
        last_t= torch.tensor([[0.0,0.0,0.0,1.0]], device ="cuda")

        intr_mat_K = torch.cat([torch.cat([intrinsics_mat, zero_t.repeat(cat.shape[0],1,1)], 2), last_t.repeat(cat.shape[0],1,1)], 1)
        
        intr_mat_inv_K = torch.inverse(intr_mat_K)
                    
        return intr_mat_K, intr_mat_inv_K