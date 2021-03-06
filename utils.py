# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
import torch
from torch import nn
from six.moves import urllib


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

# def load_model(model_name, scales=[0,1,2,3], pretrained_path:str=None):
#     """
#     Loads one of the following models
#     Inputs: 
#         model_name: resnet, fastdepth, pydnet, rexnet
#         scales:
#         pretrained_path: Path to pretrained .pth file
#     Outputs:
#         outputs: model, 
#                  encoder if 'resnet' else None
#     """
#     encoder = None
#     if model_name in ["fastdepth", "pydnet", "rexnet"]:
#         if model_name == "pydnet":
#             model = Pyddepth(scales, True, False)
#         elif model_name == "rexnet":
#             model = ReXDepth(scales)#, 
#                                     # pretrained_path=os.path.join("monodepth2", "networks", "rexnet", "rexnetv1_1.0x.pth"))
#         elif model_name == "fastdepth":
#             model = fastdepth.MobileNetSkipAddMultiScale(False,
#                             pretrained_path = "", scales = scales)

#         if pretrained_path:

#     else:
#         encoder = networks.ResnetEncoder(
#             opt.num_layers, opt.weights_init == "pretrained")
#         encoder.to(device)
        
#         num_ch_enc = encoder.module.num_ch_enc if opt.distributed else encoder.num_ch_enc
#         decoder = networks.DepthDecoder(
#                                         num_ch_enc, scales)
#         model = CombineEncoderDecoder(encoder, decoder)

    
#     return model, decoder

class CombineEncoderDecoder(nn.Module):
    """This is a wrapper to combine encoder and decoder into a single model
    Inputs: 
        inputs: encoder, decoder
    Outputs:
        outputs: decoder(encoder)
    """
    def __init__(self, encoder, decoder):
        super(CombineEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, image):
        return self.decoder(self.encoder(image))


def time_torch_model(model, input, use_cuda=True, print_time=False):
    """
    Times the forward pass of a nn.Module with the given input.
    Inputs:
        model: nn-Module
        input: torch.Tensor
        use_cuda: bool       To use CUDA GPU or not (default: True)
        print_time: bool     Print the time?        (default: False)
    Output:
        total_time_ms
    """
    model.eval()
    model.cuda() if use_cuda else None
    input.cuda() if use_cuda else None
    torch.cuda.synchronize()
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
            model(input)
    total_time_ms = sum([item.cuda_time for item in prof.function_events])/1000
    if print_time:
        print("{:.3f} ms".format(total_time_ms))
    return total_time_ms