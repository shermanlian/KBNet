import os
import sys
from decimal import Decimal
import cv2
import utility
import torchvision.utils as tvutils

import torch
from tensorboardX import SummaryWriter

from utils.postprocessing_functions import SimplePostProcess
from utils.data_format_utils import convert_dict
from utils.metrics import PSNR, L1, L2, CharbonnierLoss, MSSSIMLoss
from datasets.burstsr_dataset import pack_raw_image, flatten_raw_image_batch
from data_processing.camera_pipeline import demosaic
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast, GradScaler

train_log_dir = '../train_log/'

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
tfboard_name = exp_name + "_"
exp_train_log_dir = os.path.join(train_log_dir, exp_name)

LOG_DIR = os.path.join(exp_train_log_dir, 'logs')

# save img path
IMG_SAVE_DIR = os.path.join(exp_train_log_dir, 'img_log')
# Where to load model
LOAD_MODEL_DIR = os.path.join(exp_train_log_dir, 'models')
# Where to save new model
SAVE_MODEL_DIR = os.path.join(exp_train_log_dir, 'real_models')

# Where to save visualization images (for report)
RESULTS_DIR = os.path.join(exp_train_log_dir, 'report')

utility.mkdir(SAVE_MODEL_DIR)
utility.mkdir(IMG_SAVE_DIR)
utility.mkdir(LOG_DIR)


class Trainer():
    def __init__(self, args, train_loader, train_sampler, valid_loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale[0]

        self.ckp = ckp
        self.loader_train = train_loader
        self.loader_valid = valid_loader
        self.train_sampler = train_sampler
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        self.psnr_fn = PSNR(boundary_ignore=40)
        # Postprocessing function to obtain sRGB images
        self.postprocess_fn = SimplePostProcess(return_np=True)

        if 'L1' in args.loss:
            self.aligned_loss = L1(boundary_ignore=None).cuda(args.local_rank)
        elif 'MSE' in args.loss:
            self.aligned_loss = L2(boundary_ignore=None).cuda(args.local_rank)
        elif 'CB' in args.loss:
            self.aligned_loss = CharbonnierLoss(boundary_ignore=None).cuda(args.local_rank)
        elif 'MSSSIM' in args.loss:
            self.aligned_loss = MSSSIMLoss(boundary_ignore=None).cuda(args.local_rank)

        if self.args.fp16:
            self.scaler = GradScaler()

        self.best_psnr = 0.
        self.best_epoch = 0

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.glob_iter = 0

        self.log_dir = LOG_DIR + "/" + args.save
        self.img_save_dir = IMG_SAVE_DIR + "/" + args.save
        # Where to load model
        self.load_model_dir = LOAD_MODEL_DIR + "/" + args.save
        # Where to save new model
        self.save_model_dir = SAVE_MODEL_DIR + "/" + args.save

        # Where to save visualization images (for report)
        self.results_dir = RESULTS_DIR + "/" + args.save

        utility.mkdir(self.save_model_dir)
        utility.mkdir(self.img_save_dir)
        utility.mkdir(self.log_dir)

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        if epoch % 200 == 0:
            self.ckp.write_log(
                '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
            )
        self.loss.start_log()
        self.model.train()
        # self.test()

        if self.args.local_rank <= 0:
            timer_data, timer_model, timer_epoch = utility.timer(), utility.timer(), utility.timer()
            timer_epoch.tic()

        for batch, batch_value in enumerate(self.loader_train):

            burst, gt, kernel_gt, flow_vectors, meta_info = batch_value
            burst, gt, kernel_gt, flow_vectors = self.prepare(burst, gt, kernel_gt, flow_vectors)

            if self.args.local_rank == 0:
                timer_data.hold()
                timer_model.tic()

            if self.args.fp16:
                with autocast():
                    sr, kernel = self.model(burst, 0)
                    loss_sr = self.aligned_loss(sr, gt)
                    loss_kr = self.aligned_loss(kernel, kernel_gt) * 5.0
                    loss = loss_sr + loss_kr
            else:
                sr, kernel = self.model(burst, 0)
                loss_sr = self.aligned_loss(sr, gt)
                loss_kr =  self.aligned_loss(kernel, kernel_gt) * 5.0
                loss = loss_sr  + loss_kr

            if self.args.n_GPUs > 1:
                torch.distributed.barrier()
                reduced_loss_sr = utility.reduce_mean(loss_sr, self.args.n_GPUs)
                reduced_loss_kr = utility.reduce_mean(loss_kr, self.args.n_GPUs)

            else:
                reduced_loss = loss

            self.optimizer.zero_grad()
            if self.args.fp16:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), .05)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.model.module.Restorer.parameters(), .05)
                self.optimizer.step()

            if self.args.local_rank == 0:
                timer_model.hold()
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t[{:.4f}][{:.5f}]\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        reduced_loss_sr.item(),
                        reduced_loss_kr.item(),
                        timer_model.release(),
                        timer_data.release()))

                self.glob_iter += 1
                timer_data.tic()

        # torch.cuda.empty_cache()
        if self.args.local_rank == 0:
            timer_epoch.hold()
            print('Epoch {} cost time: {:.1f}s, lr: {:5f}'.format(epoch, timer_epoch.release(), lr))
        self.test()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.model.eval()
        if self.args.local_rank == 0:
            print("Testing...")
            timer_test = utility.timer()
        if epoch % 1 == 0:
            self.model.eval()
            total_psnr = 0
            count = 0
            # print("Testing...")
            for i, batch_value in enumerate(self.loader_valid):

                burst_, gt, kernel_gt, flow_vectors, meta_info = batch_value
                burst_, gt, kernel_gt, flow_vectors = self.prepare(burst_, gt, kernel_gt, flow_vectors)

                with torch.no_grad():
                    if self.args.fp16:
                        with autocast():
                            sr, kernel = self.model(burst_, 0)
                    else:
                        sr, kernel = self.model(burst_, 0)

                if self.args.use_tree:
                    sr = sr[0]
                score = self.psnr_fn(sr, gt)

                if self.args.n_GPUs > 1:
                    torch.distributed.barrier()
                    score = utility.reduce_mean(score, self.args.n_GPUs)

                total_psnr += score
                count += 1

            # torch.cuda.empty_cache()

            total_psnr = total_psnr / count
            if self.args.local_rank == 0:
                print("[Epoch: {}][PSNR: {:.4f}][Best PSNR: {:.4f}][Best Epoch: {}]".format(epoch, total_psnr,
                                                                                            self.best_psnr,
                                                                                            self.best_epoch))
                if epoch >= 0 and total_psnr > self.best_psnr:
                    self.best_psnr = total_psnr
                    self.best_epoch = epoch
                    filename = exp_name + 'best_epoch.pth'
                    self.save_model(filename)

                print('Forward: {:.2f}s\n'.format(timer_test.toc()))

                if (epoch) % 5 == 0 and not self.args.test_only:
                    filename = exp_name + '_epoch_' + str(epoch) + '.pth'
                    self.save_model(filename)

        torch.set_grad_enabled(True)

    def save_model(self, filename):
        print('save model...')
        net_save_path = os.path.join(self.save_model_dir, filename)
        model = self.model.model
        if self.args.n_GPUs > 1:
            model = model.module

        torch.save(model.state_dict(), net_save_path)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:{}'.format(self.args.local_rank))

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        # print(_prepare(args[0]).device)
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
