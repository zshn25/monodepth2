# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

import os
import torch.multiprocessing as mp

if __name__ == "__main__":
   
    options = MonodepthOptions()
    opts = options.parse()

    if opts.gpu: # is not None
        opts.world_size = len(opts.gpu)

        os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join(str(x) for x in opts.gpu) # "0,1"

        if opts.distributed:

            mp.spawn(Trainer,
                     args=(opts,),
                     nprocs=opts.world_size,
                     join=True)
        else:
            Trainer(None, opts)
    else:
        # CPU mode
        opts.distributed = False
        print("Training on CPU. If GPU training is expected, give --gpu ")
        Trainer(None, opts)
    
