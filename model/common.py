import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SFT(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, ext_feats, act, bias=True):

        super(SFT, self).__init__()

        self.gamma_conv = conv(n_feats + ext_feats, n_feats, kernel_size, bias=bias)
        self.beta_conv = conv(n_feats + ext_feats, n_feats, kernel_size, bias=bias)
        self.act = act

    def forward(self, x, h):
        x_h = torch.cat([x, h], dim=1)
        gamma = torch.sigmoid(self.gamma_conv(x_h))
        beta = self.beta_conv(x_h)
        res = x * gamma + beta

        return self.act(res)

class ResSFTBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, ext_feats,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResSFTBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(SFT(conv, n_feats, kernel_size, ext_feats, act=act, bias=bias))
            m.append(ResBlock(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        self.body = nn.ModuleList(m)
        self.res_scale = res_scale

    def forward(self, x, components):
        res = x
        for i in range(len(self.body)):
            if i % 2 == 0:
                res = self.body[i](res, components)
            else:
                res = self.body[i](res)

        res = res.mul(self.res_scale) + x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class UpOnly(nn.Sequential):
    def __init__(self, scale):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelShuffle(2))


        elif scale == 3:

            m.append(nn.PixelShuffle(3))

        else:
            raise NotImplementedError

        super(UpOnly, self).__init__(*m)

