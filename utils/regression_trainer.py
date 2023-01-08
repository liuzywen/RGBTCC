from utils.evaluation import eval_game, eval_relative
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
from nets.RGBTCCNet import ThermalRGBNet
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.crowd import Crowd
from losses.ot_loss import OT_Loss

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, gt_discretes, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            # assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, "new_" + x + "_224"),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  x) for x in ['train', 'val', 'test']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val', 'test']}

        self.model = ThermalRGBNet(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.ot_loss = OT_Loss(args.crop_size, self.downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot,
                               args.reg)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.count_loss = nn.L1Loss(size_average=False).to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.best_count_1 = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                mae_is_best, mse_is_best = self.val_epoch()
            if epoch >= args.val_start and (mse_is_best or mae_is_best ):#or (epoch > 200 and epoch % 5 == 0)):
                self.test_epoch()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, gt_discrete, st_sizes) in enumerate(self.dataloaders['train']):

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            if type(inputs) == list:
                N = inputs[0].size(0)
            else:
                N = inputs.size(0)

            with torch.set_grad_enabled(True):
                count, outputs, outputs_normed = self.model(inputs)
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                # Compute L1 loss.
                l1_loss = self.count_loss(count.sum(1).sum(1), torch.from_numpy(gd_count).float().to(self.device))
                loss = ot_loss + count_loss + tv_loss + l1_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_game.update(np.mean(abs(pred_err)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, GAME0: {:.2f} MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_game.get_avg(), np.sqrt(epoch_mse.get_avg()),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        self.model.eval()  # Set model to evaluate mode
        epoch_start = time.time()
        total_relative_error = 0
        epoch_res = []
        for inputs, target, name in self.dataloaders['val']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            if len(inputs[0].shape) == 5:
                inputs[0] = inputs[0].squeeze(0)
                inputs[1] = inputs[1].squeeze(0)
            if len(inputs[0].shape) == 3:
                inputs[0] = inputs[0].unsqueeze(0)
                inputs[1] = inputs[1].unsqueeze(0)

            with torch.set_grad_enabled(False):
                _, outputs, _ = self.model(inputs)
                res = torch.sum(target).item() - torch.sum(outputs).item()
                epoch_res.append(res)

                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error

        N = len(self.dataloaders['val'])
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        mae_is_best = mae < self.best_mae
        mse_is_best = 2 * mse < 2 * self.best_mse
        total_relative_error = total_relative_error / N
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Re: {:.4f},Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, total_relative_error, time.time() - epoch_start))

        # model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))

        return mae_is_best, mse_is_best

    def test_epoch(self):
        args = self.args
        self.model.eval()  # Set model to evaluate mode
        epoch_start = time.time()
        total_relative_error = 0
        epoch_res = []
        for inputs, target, name in self.dataloaders['test']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            if len(inputs[0].shape) == 5:
                inputs[0] = inputs[0].squeeze(0)
                inputs[1] = inputs[1].squeeze(0)
            if len(inputs[0].shape) == 3:
                inputs[0] = inputs[0].unsqueeze(0)
                inputs[1] = inputs[1].unsqueeze(0)

            with torch.set_grad_enabled(False):
                _, outputs, _ = self.model(inputs)
                res = torch.sum(target).item() - torch.sum(outputs).item()
                epoch_res.append(res)
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
        N = len(self.dataloaders['test'])
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        total_relative_error = total_relative_error / N
        logging.info('Epoch {} test, MSE: {:.2f} MAE: {:.2f}, Re: {:.4f},Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, total_relative_error, time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(mae)))


