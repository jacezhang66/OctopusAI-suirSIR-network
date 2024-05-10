import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
import torchvision.utils as vutils
from option import opt
from dataloader_util import *
from torchvision.models import vgg16
import sys, os
from Network_New import suirSIR
from itertools import cycle
from torch.autograd import Variable
from adamp import AdamP
from torch.optim import lr_scheduler
from utils import *
from torch.utils.tensorboard import SummaryWriter
from loss import *
from tqdm import tqdm
from torch.utils.data import DataLoader

dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)

start_time = time.time()


class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch, writer):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.final_epoch  # 200
        self.save_period = 5
        self.loss_supl1 = nn.L1Loss()
        self.loss_selfgray = GrayWorldloss()
        self.loss_ssim = SSIM()
        self.loss_selfinput = nn.L1Loss()
        self.loss_unsupl1 = nn.L1Loss()
        self.loss_selgD = nn.L1Loss()
        self.loss_t = nn.L1Loss()
        self.loss_unsupinput = nn.L1Loss()
        self.loss_unsupselfinput = nn.L1Loss()

        self.consistency = 0.2
        self.consistency_rampup = 100.0
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()

        self.loss_vgg = PerpetualLoss(vgg_model).cuda()
        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        self.optimizer_s = AdamP(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[100, 150], gamma=0.1)

        self.model_save = args.model_dir + 'model_e140.pth'
        self.teacher_pl_path = './visual/pl'
        self.teacher_raw_path = './visual/raw'
        self.teacher_td_path = './visual/td'
        self.teacher_tb_path = './visual/tb'
        self.teacher_a_path = './visual/a'
        self.teacher_s_path = './visual/s'

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul, predict_target_raw_ul, _, _, _, _ = self.tmodel(image)

        return predict_target_ul, predict_target_raw_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def get_record(self, teacher_predict, teacher_predict_raw, p_name, tpa, tra, epoch):
        N = teacher_predict.shape[0]
        for idx in range(0, N):
            id = p_name[idx].split('\\')[-1]
            path_tpa = os.path.join(tpa, str(epoch), id)
            path_tra = os.path.join(tra, str(epoch), id)
            if not os.path.exists(os.path.join(tpa, str(epoch))):
                os.mkdir(os.path.join(tpa, str(epoch)))
            if not os.path.exists(os.path.join(tra, str(epoch))):
                os.mkdir(os.path.join(tra, str(epoch)))

            p_l = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
            p_l = np.clip(p_l, 0, 1)
            p_l = (p_l * 255).astype(np.uint8)
            p_l = Image.fromarray(p_l)
            p_l.save('%s' % path_tpa)

            p_r = np.transpose(teacher_predict_raw[idx].detach().cpu().numpy(), (1, 2, 0))
            p_r = np.clip(p_r, 0, 1)
            p_r = (p_r * 255).astype(np.uint8)
            p_r = Image.fromarray(p_r)
            p_r.save('%s' % path_tra)

        del N, teacher_predict, teacher_predict_raw

        return p_l

    def get_record2(self, td, tb, s, a, p_name, dpath, bpath, apath, spath, epoch):
        N = td.shape[0]
        for idx in range(0, N):
            id = p_name[idx].split('\\')[-1]
            path_td = os.path.join(dpath, str(epoch), id)
            path_tb = os.path.join(bpath, str(epoch), id)
            path_a = os.path.join(apath, str(epoch), id)
            path_s = os.path.join(spath, str(epoch), id)
            if not os.path.exists(os.path.join(dpath, str(epoch))):
                os.mkdir(os.path.join(dpath, str(epoch)))
            if not os.path.exists(os.path.join(bpath, str(epoch))):
                os.mkdir(os.path.join(bpath, str(epoch)))
            if not os.path.exists(os.path.join(apath, str(epoch))):
                os.mkdir(os.path.join(apath, str(epoch)))
            if not os.path.exists(os.path.join(spath, str(epoch))):
                os.mkdir(os.path.join(spath, str(epoch)))

            p_td = np.transpose(td[idx].detach().cpu().numpy(), (1, 2, 0))
            p_td = np.clip(p_td, 0, 1)
            p_td = (p_td * 255).astype(np.uint8)
            p_td = Image.fromarray(p_td)
            p_td.save('%s' % path_td)

            p_tb = np.transpose(tb[idx].detach().cpu().numpy(), (1, 2, 0))
            p_tb = np.clip(p_tb, 0, 1)
            p_tb = (p_tb * 255).astype(np.uint8)
            p_tb = Image.fromarray(p_tb)
            p_tb.save('%s' % path_tb)

            p_r = np.transpose(a[idx].detach().cpu().numpy(), (1, 2, 0))
            p_r = np.clip(p_r, 0, 1)
            p_r = (p_r * 255).astype(np.uint8)
            p_r = Image.fromarray(p_r)
            p_r.save('%s' % path_a)

            p_s = np.transpose(s[idx].detach().cpu().numpy(), (1, 2, 0))
            p_s = np.clip(p_s, 0, 1)
            p_s = (p_s * 255).astype(np.uint8)
            p_s = Image.fromarray(p_s)
            p_s.save('%s' % path_s)

        del N, td, tb, a, s

        return p_r

    def train(self):
        self.freeze_teachers_parameters()
        if opt.resume and os.path.exists(self.args.model_dir):
            print(f'resume from {self.args.model_dir}')
            checkpoint = torch.load(self.model_save)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f'train from %s', self.start_epoch)
        else:
            initialize_weights(self.model)
            print(f'train from %s', self.start_epoch)

        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_ave, psnr_train = self._train_epoch(epoch)
            loss_val = loss_ave.item() / self.args.crop_size * self.args.bs
            train_psnr = sum(psnr_train) / len(psnr_train)
            psnr_val = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)

            print('[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f' % (
                epoch, loss_val, train_psnr, val_psnr, self.lr_scheduler_s.get_last_lr()[0]))

            if epoch % self.save_period == 0:
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                ckpt_name = str(self.args.model_dir) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

    def _train_epoch(self, epoch):
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        sup1 = AverageMeter()
        sup2 = AverageMeter()
        sup3 = AverageMeter()
        sup4 = AverageMeter()
        sup5 = AverageMeter()
        unsup1 = AverageMeter()
        unsup2 = AverageMeter()
        unsup3 = AverageMeter()
        unsup4 = AverageMeter()
        unsup5 = AverageMeter()

        loss_total_ave = 0.0
        psnr_train = []
        self.model.train()
        self.freeze_teachers_parameters()
        train_loader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = range(len(self.unsupervised_loader))
        tbar = tqdm(tbar, ncols=130, leave=True)
        for i in tbar:
            (img_data, label, T1), (unpaired_data_w, unpaired_data_s, name) = next(train_loader)
            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            T1 = Variable(T1).cuda(non_blocking=True)
            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)
            unpaired_data_w = Variable(unpaired_data_w).cuda(non_blocking=True)
            predict_target_u, predict_target_raw_u = self.predict_with_out_grad(unpaired_data_w)  # 生成伪标签,raw
            origin_predict = predict_target_u.detach().clone()

            tpa = self.teacher_pl_path
            tra = self.teacher_raw_path
            if epoch % self.save_period == 0:
                yss = self.get_record(predict_target_u, predict_target_raw_u, name, tpa, tra, epoch)

            outputs_l, raw_l, TD, TB, s, A = self.model(img_data)  # 有标签输入的输出
            outputs_ul, raw_ul, TD_ul, TB_ul, s, A = self.model(unpaired_data_s)  # 无标签输入的输出

            dpath = self.teacher_td_path = './visual/td'
            bpath = self.teacher_tb_path = './visual/tb'
            apath = self.teacher_a_path = './visual/a'
            spath = self.teacher_a_path = './visual/s'
            if epoch % self.save_period == 0:
                pxx = self.get_record2(TD, TB, s, A, name, dpath, bpath, apath, spath, epoch)

            loss_selfinput_raw = 0.5 * self.loss_selfinput(raw_l, img_data)
            loss_trans_score = self.loss_selgD(TD, TB)
            loss_supl1_score = 2 * self.loss_supl1(outputs_l, label)
            loss_selfgray_score = 0.2 * self.loss_selfgray(outputs_l)
            loss_t = self.loss_t(TD, T1)
            loss_sup = loss_supl1_score + loss_selfgray_score + loss_selfinput_raw + loss_trans_score + loss_t
            sup_loss.update(loss_sup.mean().item())
            sup1.update(loss_supl1_score.mean().item())
            sup2.update(loss_selfgray_score.mean().item())
            sup3.update(loss_selfinput_raw.mean().item())
            sup4.update(loss_trans_score.mean().item())
            sup5.update(loss_t.mean().item())

            loss_unsupl1_score = 2 * self.loss_unsupl1(outputs_ul, predict_target_u)
            loss_unsuptrans_score = self.loss_selgD(TD_ul, TB_ul)
            loss_unsupselfinput_raw_s = 0.5 * self.loss_unsupselfinput(raw_ul, unpaired_data_s)
            loss_unsupssim = 0.2 * self.loss_ssim(raw_ul, predict_target_raw_u)
            loss_unsu = loss_unsupl1_score + loss_unsuptrans_score + loss_unsupselfinput_raw_s + loss_unsupssim

            unsup_loss.update(loss_unsu.mean().item())
            unsup1.update(loss_unsupl1_score.mean().item())
            unsup3.update(loss_unsuptrans_score.mean().item())
            unsup4.update(loss_unsupselfinput_raw_s.mean().item())
            unsup5.update(loss_unsupssim.mean().item())

            consistency_weight = self.get_current_consistency_weight(epoch)  # 动态学习loss策略
            total_loss = consistency_weight * loss_unsu + loss_sup
            total_loss = total_loss.mean()
            psnr_train.extend(to_psnr(outputs_l, label))
            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description(
                'Train-Student Epoch {} | Ls {:.4f} Lu {:.4f}| {:.2f},{:.2f},{:.2f},{:.2f},{:.2f} | {:.2f},{:.2f},{:.2f},{:.2f}'
                    .format(epoch, sup_loss.avg, unsup_loss.avg, sup1.avg, sup2.avg, sup3.avg, sup4.avg, sup5.avg,
                            unsup1.avg, unsup3.avg, unsup4.avg, unsup5.avg))

            del img_data, label, unpaired_data_w, unpaired_data_s
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        loss_total_ave = loss_total_ave + total_loss

        self.writer.add_scalar('Train_loss', total_loss, global_step=epoch)
        self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
        self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)

        self.writer.add_scalar('sup1', sup1.avg, global_step=epoch)
        self.writer.add_scalar('sup2', sup2.avg, global_step=epoch)
        self.writer.add_scalar('sup3', sup3.avg, global_step=epoch)
        self.writer.add_scalar('sup4', sup4.avg, global_step=epoch)
        self.writer.add_scalar('sup5', sup5.avg, global_step=epoch)
        self.writer.add_scalar('unsup1', unsup1.avg, global_step=epoch)
        self.writer.add_scalar('unsup3', unsup3.avg, global_step=epoch)
        self.writer.add_scalar('unsup4', unsup4.avg, global_step=epoch)
        self.writer.add_scalar('unsup5', unsup5.avg, global_step=epoch)
        self.lr_scheduler_s.step(epoch=epoch - 1)
        return loss_total_ave, psnr_train

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.tmodel.eval()
        val_psnr = AverageMeter()
        total_loss_val = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data, val_label) in enumerate(tbar):
                val_data = Variable(val_data).cuda()
                val_label = Variable(val_label).cuda()
                val_output, _, _, _, _, _ = self.model(val_data)
                temp_psnr, N = compute_psnr_ssim(val_output, val_label)
                val_psnr.update(temp_psnr, N)
                psnr_val.extend(to_psnr(val_output, val_label))
                tbar.set_description(
                    '{} Epoch {} | PSNR: {:.4f}|'.format("Eval-Student", epoch, val_psnr.avg))

            self.writer.add_scalar('Val_psnr', val_psnr.avg, global_step=epoch)
            del val_output, val_label, val_data
            return psnr_val

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):

        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


if __name__ == '__main__':
    setup_seed(2022)
    train_folder = opt.train_folder
    paired_dataset = TrainLabeled(dataroot=train_folder, phase='labled1',
                                  finesize=opt.crop_size)
    unpaired_dataset = TrainUnlabeled(dataroot=train_folder, phase='unlabeled',
                                      finesize=opt.crop_size)
    val_dataset = ValLabeled(dataroot=train_folder, phase='val', finesize=opt.crop_size)
    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None
    paired_loader = DataLoader(paired_dataset, batch_size=opt.bs, sampler=paired_sampler)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=opt.testbs, sampler=unpaired_sampler)
    val_loader = DataLoader(val_dataset, batch_size=opt.testbs, sampler=val_sampler)
    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))
    net = suirSIR()
    ema_net = suirSIR()
    ema_net = create_emamodel(ema_net)
    writer = SummaryWriter(log_dir=opt.log_dir)
    trainer = Trainer(model=net, tmodel=ema_net, args=opt, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer)
    trainer.train()
    writer.close()
