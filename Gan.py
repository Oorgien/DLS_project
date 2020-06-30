from torch import *

import gc

import torch
import torch.nn as nn

from Images import Preproc, Normalization

import pytorch_CycleGAN_and_pix2pix.models.networks as net
import functools

from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.models import create_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class GanModel(nn.Module):
    # normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    # normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization_mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    normalization_std = torch.tensor([0.5, 0.5, 0.5]).to(device)

    def __init__(self, model_name):
        super(GanModel, self).__init__()
        self.model_name = './pytorch_CycleGAN_and_pix2pix/checkpoints/' + model_name + '/latest_net_G.pth'
        self.model_jit_name = './pytorch_CycleGAN_and_pix2pix/checkpoints/' + model_name + '/latest_net_G.jit'
        self.model = self.load_model()

    def get_model(self):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        model = net.ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        model.load_state_dict(torch.load(self.model_name))
        model = model.cuda()
        model = model.eval()
        return model
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # torch.backends.cudnn.benchmark = True

    def load_model(self):
        model_file = self.model_jit_name
        model = torch.jit.load(model_file).cuda()
        model = model.eval()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.backends.cudnn.benchmark = True
        return model

    def forward(self, img, img_size):

        normalization = Normalization(self.normalization_mean, self.normalization_std).to(device)
        content_img = Preproc(img_size).image_loader(img)
        normalized_img = normalization.forward(content_img)

        gc.collect()
        with torch.no_grad():
            res = self.model(normalized_img).detach()
        res = res.view(res.shape[1], res.shape[2], res.shape[3])
        res = (res * self.normalization_std.view(3, 1, 1)) + self.normalization_mean.view(3, 1, 1)
        return res



def GanScrypt(img_path, model_name, imsize):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on
    # randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    opt.dataroot = ''
    opt.name = model_name
    opt.model = 'test'
    opt.no_droput = True

    opt.crop_size = imsize
    opt.load_size = imsize

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    img = Preproc(imsize).image_loader(img_path)
    data = {'A': img, 'A_paths': ''}
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image results
    normalization_mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    normalization_std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    return visuals['fake'] * normalization_mean.view(-1, 1, 1) + normalization_std.view(-1, 1, 1)



