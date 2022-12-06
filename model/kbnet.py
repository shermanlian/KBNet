''' network architecture for EDVR '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.arch_util as arch_util
import model.common as common
# from torch.cuda.amp import autocast

import torchvision.utils as tvutils
from utils import PCAEncoder

try:
    from model.DCNv2.dcn_v2 import DCN_sep, DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


def make_model(args, parent=False):
    return KBNet(args)

class Estimator(nn.Module):
    def __init__(
        self, args
    ):
        super(Estimator, self).__init__()

        in_nc=args.burst_channel
        nf=args.n_feats//2
        num_blocks=3
        self.ksize = args.kernel_size

        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 5, 1, 2),
            arch_util.make_layer(basic_block, num_blocks)
        )

        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, self.ksize ** 2, 1, 1, 0),
            nn.Softmax(1),
        )

    def forward(self, LR):
        f = self.head(LR)
        f = self.body(f)
        f = self.tail(f)

        return f.view(f.shape[0], 1, self.ksize, self.ksize)

class KAPCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

    def __init__(self, nf=64, groups=8, ker_code_length=10, wn=None):
        super(KAPCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = wn(nn.Conv2d(nf*2+ker_code_length*2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L3_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = wn(nn.Conv2d(nf*2+ker_code_length*2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L2_offset_conv2 = wn(nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True))  # concat for offset
        self.L2_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = wn(nn.Conv2d(nf*2+ker_code_length*2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L1_offset_conv2 = wn(nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True))  # concat for offset
        self.L1_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.L1_fea_conv = wn(nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True))  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = wn(nn.Conv2d(nf*2+ker_code_length, nf, 3, 1, 1, bias=True))  # concat for diff
        self.cas_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l, nbr_ker, ref_ker):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        B_h, C_h = nbr_ker.size()

        # L3
        _, _, H3, W3 = nbr_fea_l[2].size()
        ref_ker_exp3 = ref_ker.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H3, W3))  # kernel_map stretch
        nbr_ker_exp3 = nbr_ker.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H3, W3))  # kernel_map stretch
        L3_offset = torch.cat([nbr_fea_l[2], nbr_ker_exp3, ref_fea_l[2], ref_ker_exp3], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        _, _, H2, W2 = nbr_fea_l[1].size()
        ref_ker_exp2 = ref_ker.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H2, W2))  # kernel_map stretch
        nbr_ker_exp2 = nbr_ker.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H2, W2))  # kernel_map stretch
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = torch.cat([nbr_fea_l[1], nbr_ker_exp2, ref_fea_l[1], ref_ker_exp2], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))

        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        _, _, H1, W1 = nbr_fea_l[0].size()
        ref_ker_exp1 = ref_ker.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H1, W1))  # kernel_map stretch
        nbr_ker_exp1 = nbr_ker.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H1, W1))  # kernel_map stretch
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = torch.cat([nbr_fea_l[0], nbr_ker_exp1, ref_fea_l[0], ref_ker_exp1], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))

        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0], ref_ker_exp1], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea


class Restorer(nn.Module):
    def __init__(self, args):
        super(Restorer, self).__init__()
        nf = args.n_feats
        n_resblocks = args.n_resblocks
        nframes = args.burst_size
        front_RBs=5
        back_RBs=args.n_resgroups #20
        groups=8
        code_length = args.ker_code_length
        self.ksize = args.kernel_size

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.center = 0

        bRB = functools.partial(arch_util.ResidualGroup, n_feat=nf, n_resblocks=n_resblocks)

        #### extract features (for each frame)
        self.conv_first = wn(nn.Conv2d(args.burst_channel*1, nf, 3, 1, 1, bias=True))
        self.feature_extraction = nn.ModuleList([
            common.ResSFTBlock(common.default_conv, nf, 3, code_length) for _ in range(front_RBs)
        ])
        self.fea_L2_conv1 = wn(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        self.fea_L3_conv1 = wn(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))

        ################################################################
        self.pcd_align = PCD_Align(nf=nf, groups=groups, ker_code_length=code_length, wn=wn)
        #################################################################

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(bRB, back_RBs)

        #### upsampling
        self.upconv1 = wn(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        self.upconv2 = wn(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = wn(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.conv_last = wn(nn.Conv2d(64, args.n_colors, 3, 1, 1, bias=True))

        #### activation function
        self.lrelu = nn.ReLU(inplace=True)

    # @autocast()
    def forward(self, x, ker_map):
        B, N, C, H, W = x.size()  # N video frames
        ker_map = ker_map.view(B*N, -1)
        x_center = x[:, self.center, :, :, :].contiguous()
        kernel_maps = torch.ones((B*N, ker_map.shape[-1], H, W)).to(x.device)
        for b in range(B*N):
            for i in range(ker_map.shape[-1]):
                kernel_maps[b, i] *= ker_map[b, i]

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        for i in range(len(self.feature_extraction)):
            L1_fea = self.feature_extraction[i](L1_fea, kernel_maps)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### kernel guided PCD align
        ker_maps = ker_map.view(B, N, -1)
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]

        ref_ker = ker_maps[:, self.center]

        aligned_fea = []

        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            nbr_ker = ker_maps[:, i]

            fea = self.pcd_align(nbr_fea_l, ref_fea_l, nbr_ker, ref_ker)
            aligned_fea.append(fea)

        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

        fea = torch.mean(aligned_fea, dim=1)

        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out


class KBNet(nn.Module):
    def __init__(self, args):
        super(KBNet, self).__init__()

        self.ksize = args.kernel_size
        self.Restorer = Restorer(args)
        self.Estimator = Estimator(args)

        pca_matrix_path = "./pca_matrix/pca_aniso_matrix_x4.pth"
        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

    # @autocast()
    def forward(self, lr):
        B, N, C, H, W = lr.shape

        lr_reshape = lr.view(B*N, C, H, W)
        kernel = self.Estimator(lr_reshape).view(B, N, self.ksize, self.ksize)
        ker_map = kernel.view(B*N, 1, self.ksize ** 2).matmul(self.encoder)[:, 0]
        sr = self.Restorer(lr, ker_map)

        return sr, kernel

