
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import utility
from option import args
import torchvision.utils as tvutils

from datasets.synthetic_burst_val_set import SyntheticBurstVal
from datasets.burstsr_dataset import flatten_raw_image_batch, pack_raw_image
from utils.metrics import PSNR, SSIM, LPIPS
from utils.data_format_utils import convert_dict
from data_processing.camera_pipeline import demosaic
from utils.postprocessing_functions import SimplePostProcess
import model

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import time


checkpoint = utility.checkpoint(args)


def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))


def main_worker(local_rank, nprocs, args):

    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    print(f'============  burst size {args.burst_size}  =============')
    dataset = SyntheticBurstVal(args.root, args.burst_size)
    out_dir = f'val/kbnet/{args.root.split("/")[-1]}'

    _model = model.Model(args, checkpoint)

    for param in _model.parameters():
        param.requires_grad = False

    boundary_ignore = 40
    psnr_fn = PSNR(boundary_ignore=boundary_ignore)
    ssim_fn = SSIM(boundary_ignore=boundary_ignore)
    lpips_fn = LPIPS(boundary_ignore=boundary_ignore)

    postprocess_fn = SimplePostProcess(return_np=True)

    os.makedirs(out_dir, exist_ok=True)

    tt = []
    psnrs, ssims, lpipss = [], [], []
    for idx in tqdm(range(len(dataset))):
        burst, gt, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst_ = burst.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()
        burst_ = flatten_raw_image_batch(burst_)

        with torch.no_grad():
            tic = time.time()
            sr, kernel = _model(burst_, 0)
            sr = sr.float()
            toc = time.time()
            tt.append(toc-tic)

            psnr = psnr_fn(sr, gt)
            ssim = ssim_fn(sr, gt)
            lpips = lpips_fn(sr.contiguous(), gt)
            psnrs.append(psnr.item())
            ssims.append(ssim.item())
            lpipss.append(lpips.item())


    print(f'avg PSNR: {np.mean(psnrs):.6f}')
    print(f'avg SSIM: {np.mean(ssims):.6f}')
    print(f'avg LPIPS: {np.mean(lpipss):.6f}')
    print(f' avg time: {np.mean(tt):.6f}')


if __name__ == '__main__':
    main()
