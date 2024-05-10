import torch
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from option import *


def lr_schedule_cosdecay(t, T=opt.steps, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):

        nn.init.kaiming_normal(m.weight.data, mode='fan_out')

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)

    return psnr / recoverd.shape[0], recoverd.shape[0]
