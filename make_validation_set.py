import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.utils as tvutils

import utility
import model
import loss
from option import args
from trainer import Trainer
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.synthetic_burst_val_set import SyntheticBurstVal
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import pickle as pkl
import os
import cv2

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


checkpoint = utility.checkpoint(args)


def main():
    args.n_GPUs = 1
    if args.n_GPUs > 1:
        mp.spawn(main_worker, nprocs=args.n_GPUs, args=(args.n_GPUs, args))
    else:
        main_worker(0, args.n_GPUs, args)


def main_worker(local_rank, nprocs, args):
    if checkpoint.ok:
        args.local_rank = local_rank
        if nprocs > 1:
            init_seeds(local_rank+1)
            cudnn.benchmark = True
            utility.setup(local_rank, nprocs)
        torch.cuda.set_device(args.local_rank)

        batch_size = int(args.batch_size / nprocs)

        valid_zurich_raw2rgb = ZurichRAW2RGB(root=args.root, split='test')
        valid_data = SyntheticBurst(valid_zurich_raw2rgb, burst_size=args.burst_size, crop_sz=384)

        # root = '../test_set_0_1.6'
        # root = '../test_set_1.6_3.2'
        root = '../test_set_3.2_4.8'
        print(root)
        burst_dir = os.path.join(root, 'bursts')
        gt_dir = os.path.join(root, 'gt')

        utility.mkdir(root)
        utility.mkdir(burst_dir)
        utility.mkdir(gt_dir)

        for i, batch_value in enumerate(valid_data):
            print(i)
            utility.mkdir(f'{gt_dir}/{i:04d}')
            burst, gt, kernel_gt, flow_vectors, meta_info = batch_value
            gt_ = (gt.permute(1, 2, 0) * 2**14).numpy().astype(np.uint16)
            cv2.imwrite(f'{gt_dir}/{i:04d}/im_rgb.png', gt_)

            with open(f'{gt_dir}/{i:04d}/meta_info.pkl', 'wb') as handle:
                pkl.dump(meta_info, handle, protocol=pkl.HIGHEST_PROTOCOL)

            for n, burst_im in enumerate(burst):
                utility.mkdir(f'{burst_dir}/{i:04d}')
                im_ = (burst_im.permute(1, 2, 0) * 2**14).numpy().astype(np.uint16)
                cv2.imwrite(f'{burst_dir}/{i:04d}/im_raw_{n:02d}.png', im_)
                tvutils.save_image(
                    kernel_gt[n].unsqueeze(0).unsqueeze(0).data,
                    f'{burst_dir}/{i:04d}/kernel_{n:02d}.png', normalize=True)



        utility.cleanup()

        checkpoint.done()


if __name__ == '__main__':
    main()
