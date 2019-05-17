# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# Vassilis Choutas <https://ps.is.tuebingen.mpg.de/employees/vchoutas> for ContinousRotReprDecoder
#
# 2018.01.02

'''
A human body pose prior built with Auto-Encoding Variational Bayes
'''

__all__ = ['VPoser']

import os, sys, shutil

import torch

from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from datetime import datetime
from configer import Configer

from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torchgeometry as tgm
from human_body_prior.tools.omni_tools import makepath

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class VPoser(nn.Module):
    NUM_JOINTS = 21
    def __init__(self, num_neurons, latentD, data_shape, use_cont_repr=True):
        super(VPoser, self).__init__()

        self.latentD = latentD
        self.use_cont_repr = use_cont_repr

        n_features = np.prod([1, VPoser.NUM_JOINTS, 9])

        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=.25, inplace=False)

        if use_cont_repr:
            rot_dim = 3
            n_features = int(VPoser.NUM_JOINTS) * (rot_dim ** 2 - rot_dim)

        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_out = nn.Linear(num_neurons, n_features)

    def encode(self, Pin):
        '''

        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        Xout = Pin.view(Pin.size(0), -1)  # flatten input
        Xout = F.leaky_relu(self.bodyprior_enc_fc1(Xout), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc2(Xout), negative_slope=.2)
        return torch.distributions.normal.Normal(self.bodyprior_enc_mu(Xout),
                                                 F.softplus(self.bodyprior_enc_logvar(Xout)))

    def decode(self, Zin, output_type='matrot'):
        assert output_type in ['matrot', 'aa']

        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=.2)
        Xout = self.bodyprior_dec_out(Xout)
        if self.use_cont_repr:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)

        if output_type == 'aa': return VPoser.matrot2aa(Xout.view([-1, 1, VPoser.NUM_JOINTS, 9]))
        return Xout.view([-1, 1, VPoser.NUM_JOINTS, 9])

    def forward(self, Pin, input_type='matrot', output_type='matrot'):
        '''

        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert output_type in ['matrot', 'aa']
        if input_type == 'aa': Pin = VPoser.aa2matrot(Pin)
        q_z = self.encode(Pin)
        q_z_sample = q_z.rsample()
        Prec = self.decode(q_z_sample)
        if output_type == 'aa': Prec = VPoser.matrot2aa(Prec)
        return Prec, q_z

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        np.random.seed(seed)
        dtype = self.bodyprior_dec_fc1.weight.dtype
        device = self.bodyprior_dec_fc1.weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype).to(device)
        return self.decode(Zgen, output_type=output_type)

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose_aa = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose_aa

    @staticmethod
    def aa2matrot(pose_aa):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose_aa.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose_aa.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        return pose_body_matrot


class vposer_trainer:

    def __init__(self, work_dir, ps):
        from human_body_prior.data.dataloader import AMASSDataset
        from torch.utils.data import DataLoader
        from tensorboardX import SummaryWriter

        from human_body_prior.tools.omni_tools import log2file, makepath

        self.pt_dtype = torch.float64 if ps.fp_precision == '64' else torch.float32

        torch.manual_seed(ps.seed)

        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        ps.work_dir = makepath(work_dir, isfile=False)

        logger = log2file(os.path.join(work_dir, '[%s]_%s.log' % (expr_code, log_name)))

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        shutil.copy2(os.path.basename(sys.argv[0]), work_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda: torch.cuda.empty_cache()
        self.comp_device = torch.device("cuda:%d"%ps.cuda_id if torch.cuda.is_available() else "cpu")

        logger('%d CUDAs available!' % torch.cuda.device_count())

        gpu_brand= torch.cuda.get_device_name(ps.cuda_id) if use_cuda else None
        logger('Training on %s [%s]' % (self.comp_device,gpu_brand)  if use_cuda else 'Training on CPU!!!')

        # if ps.ip_avoid:
        #     from tools_torch.smpl_pt import BodyInterpenetration
        #     from nima.tools_torch.smpl_pt import BodyModel
        #     bm = BodyModel(model_pklpath=ps.ip_bmodel, model_type=ps.model_type, batch_size=ps.batch_size).to('cuda')
        #     self.body_interpenetration = BodyInterpenetration(bm)

        # kwargs = {'num_workers': ps.n_workers}
        kwargs = {'num_workers': ps.n_workers, 'pin_memory': True} if use_cuda else {'num_workers': ps.n_workers}
        ds_train = AMASSDataset(dataset_dir=os.path.join(ps.dataset_dir, 'train'))
        self.ds_train = DataLoader(ds_train, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_val = AMASSDataset(dataset_dir=os.path.join(ps.dataset_dir, 'vald'))
        self.ds_val = DataLoader(ds_val, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        logger('Train dataset size %.2f M' % (len(self.ds_train.dataset)*1e-6))
        logger('Validation dataset size %d' % len(self.ds_val.dataset))

        ps.data_shape = list(ds_val[0]['pose_matrot'].shape)
        self.vposer_model = VPoser(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape,
                                   use_cont_repr=ps.use_cont_repr).to(self.comp_device)

        enc_varlist = [var[1] for var in self.vposer_model.named_parameters() if 'bodyprior_enc' in var[0]]
        dec_varlist = [var[1] for var in self.vposer_model.named_parameters() if 'bodyprior_dec' in var[0]]

        enc_params_count = sum(p.numel() for p in enc_varlist if p.requires_grad)
        dec_params_count = sum(p.numel() for p in dec_varlist if p.requires_grad)
        logger('Encoder Trainable Parameters Count %2.2f M and in Decoder: %2.2f M.' % (
        enc_params_count * 1e-6, dec_params_count * 1e-6,))
        logger('Total Trainable Parameters Count is %2.2f M.' % ((dec_params_count + enc_params_count) * 1e-6))

        self.optimizer = optim.Adam(enc_varlist + dec_varlist, betas=(ps.adam_beta1, 0.999), lr=ps.base_lr, weight_decay=ps.reg_coef)

        self.logger = logger
        self.best_loss_total = np.inf
        self.best_model_fname = None
        self.try_num = ps.try_num
        self.epochs_completed = 0
        self.ps = ps

        chose_ids = np.random.choice(list(range(len(ds_val))), size=ps.num_bodies_to_display, replace=False, p=None)
        data_all = {}
        for id in chose_ids:
            for k, v in ds_val[id].items():
                if k in data_all.keys():
                    data_all[k] = torch.cat([data_all[k], v[np.newaxis]], dim=0)
                else:
                    data_all[k] = v[np.newaxis]

        self.vis_porig = {k: data_all[k].to(self.comp_device) for k in data_all.keys()}

        from human_body_prior.body_model.body_model import BodyModel
        bm_path = '/ps/project/common/moshpp/smplh/locked_head/female/model.npz'
        self.bm = BodyModel(bm_path, 'smplh', num_betas=16).to(self.comp_device)

        # self.swriter.add_graph(self.vposer_model, self.vis_porig, True)

    def train(self):
        self.vposer_model.train()
        save_every_it = len(self.ds_train) / self.ps.log_every_epoch
        train_loss_dict = {}
        for it, data in enumerate(self.ds_train):

            porig = data['pose_matrot'].to(self.comp_device)#.view(-1,1,22,9)#[:, :, 1:22].to(self.comp_device)  # remove root orientations and fingers
            self.optimizer.zero_grad()
            prec, q_z = self.vposer_model(porig, input_type='matrot', output_type='matrot')
            loss_total, cur_loss_dict = self.compute_loss(porig, prec, q_z)
            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = vposer_trainer.creat_loss_message(cur_train_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=it,
                                                              try_num=self.try_num, mode='train')

                self.logger(train_msg)
                self.swriter.add_histogram('q_z_sample', c2c(q_z.rsample()), it)

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self):
        self.vposer_model.eval()
        eval_loss_dict = {}
        with torch.no_grad():
            for data in self.ds_val:
                porig = data['pose_matrot'].to(self.comp_device)  # remove root orientations and fingers
                prec, q_z = self.vposer_model(porig, input_type='matrot', output_type='matrot')
                _, cur_loss_dict = self.compute_loss(porig, prec, q_z)
                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(self.ds_val) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, porig, prec, q_z):
        n_joints = VPoser.NUM_JOINTS
        batch_size = porig.size(0)
        device = porig.device
        dtype = porig.dtype
        latentD = q_z.mean.size(1)

        loss_rec = (1. - self.ps.kl_coef) * torch.mean(torch.sum(torch.pow(porig - prec, 2), dim=[1, 2, 3]))

        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl = self.ps.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        # R = prec.view([batch_size, n_joints, 3, 3])
        # R_T = torch.transpose(R, 2, 3)
        # R_eye = torch.tensor(np.tile(np.eye(3,3).reshape(1,1,3,3), [batch_size, n_joints, 1, 1]), dtype=dtype, requires_grad = False).to(device)
        # loss_ortho = self.ps.ortho_coef * torch.mean(torch.sum(torch.pow(torch.matmul(R, R_T) - R_eye,2),dim=[1,2,3]))
        #
        # det_R = torch.transpose(torch.stack([determinant_3d(R[:,jIdx,...]) for jIdx in range(n_joints)]),0,1)
        #
        # one = torch.tensor(np.ones([batch_size, n_joints]), dtype = dtype, requires_grad = False).to(device)
        # loss_det1 = self.ps.det1_coef * torch.mean(torch.sum(torch.abs(det_R - one), dim=[1]))

        loss_dict = {'loss_kl': loss_kl,
                     'loss_rec': loss_rec,
                     # 'loss_ortho':loss_ortho,
                     # 'loss_det1':loss_det1,
                     }
        if self.ps.ip_avoid:
            pose_aa = VPoser.matrot2aa(prec).view(batch_size, -1)
            bm_forwarded = self.body_interpenetration.bm(pose_body=pose_aa)
            loss_dict['loss_ip_rec'] = self.ps.ip_coef * torch.mean(
                self.body_interpenetration(bm_forwarded=bm_forwarded))

            # pgen = self.vposer_model.decode(q_z.mean, aa_out=True).view(batch_size, -1)
            # bm_forwarded = self.body_interpenetration.bm(pose_body = pgen)
            # loss_dict['loss_ip_gen'] = self.ps.ip_coef * torch.mean(self.body_interpenetration(bm_forwarded = bm_forwarded))

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def perform_training(self, num_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if num_epochs is None: num_epochs = self.ps.num_epochs

        self.logger(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), num_epochs))
        if message is not None: self.logger(expr_message)

        prev_lr = np.inf
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(num_epochs // 3), gamma=0.5)
        for epoch_num in range(1, num_epochs + 1):
            scheduler.step()
            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                self.logger('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            self.epochs_completed += 1
            train_loss_dict = self.train()
            eval_loss_dict = self.evaluate()

            with torch.no_grad():
                eval_msg = vposer_trainer.creat_loss_message(eval_loss_dict, expr_code=self.ps.expr_code,
                                                             epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                             try_num=self.try_num, mode='evald')
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.best_model_fname = makepath(os.path.join(self.ps.work_dir, 'snapshots', 'TR%02d_E%03d.pt' % (
                    self.try_num, self.epochs_completed)), isfile=True)
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.vposer_model.state_dict(), self.best_model_fname)

                    imgname = '[%s]_TR%02d_E%03d.png' % (self.ps.expr_code, self.try_num, self.epochs_completed)
                    imgpath = os.path.join(self.ps.work_dir, 'images', imgname)
                    vposer_trainer.vis_results(self.vis_porig, self.vposer_model, imgpath=imgpath, bm=self.bm)

                else:
                    self.logger(eval_msg)

                # prec, _ = self.vposer_model(self.vis_porig)
                # R = prec.view([-1, 23, 3, 3])
                # det_R = torch.transpose(torch.stack([determinant_3d(R[:, jIdx, ...]) for jIdx in range(23)]), 0, 1)
                # self.swriter.add_histogram('det_R', c2c(det_R), epoch_num)

                self.swriter.add_scalars('train_loss/scalars', train_loss_dict, self.epochs_completed)
                self.swriter.add_scalars('eval_loss/scalars', eval_loss_dict, self.epochs_completed)
                self.swriter.add_scalars('total_loss/scalars', {'train_loss_total': train_loss_dict['loss_total'],
                                                                'evald_loss_total': eval_loss_dict['loss_total'], },
                                         self.epochs_completed)

        endtime = datetime.now().replace(microsecond=0)
        self.logger(expr_message)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        self.logger('Best model path: %s\n' % self.best_model_fname)

    @staticmethod
    def creat_loss_message(loss_dict, expr_code='XX', epoch_num=0, it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s: [T:%.2e] - [%s]' % (
        expr_code, try_num, epoch_num, it, mode, loss_dict['loss_total'], ext_msg)

    @staticmethod
    def vis_results(dorig, vposer_model, imgpath, bm):
        from human_body_prior.tools.visualization_tools import render_smpl_params, imagearray2file
        from human_body_prior.train.vposer_smpl_pt import VPoser

        num_bodies_to_display = dorig['pose_aa'].size(0)
        with torch.no_grad():
            porig_aa = dorig['pose_aa']#.view([-1,1, VPoser.NUM_JOINTS, 3])

            prec_aa, q_z = vposer_model(porig_aa, input_type='aa', output_type='aa')
            pgen_aa = vposer_model.sample_poses(num_poses=num_bodies_to_display, output_type='aa')

            img_orig = render_smpl_params(bm, pose_body=porig_aa)
            img_rec = render_smpl_params(bm,  pose_body=prec_aa)
            img_gen = render_smpl_params(bm,  pose_body=pgen_aa)

            img_array = np.array([img_orig, img_rec, img_gen])
            img_array = img_array.reshape([3,num_bodies_to_display,1,400,400,3])
            imagearray2file(img_array, imgpath)


if __name__ == '__main__':

    expr_code = '0020_06_cmu_T3'
    model_type = 'smpl'

    default_ps_fname = 'vposer_defaults.ini'

    base_dir = '/ps/project/smplbodyprior/BodyPrior'

    work_dir = os.path.join(base_dir, 'VPoser', model_type, 'pytorch', expr_code)

    params = {
        'model_type': model_type,
        'batch_size': 256,
        'latentD': 32,
        'num_neurons': 512,
        'n_workers': 5,
        'cuda_id':1,

        'reg_coef': 5e-4,
        'kl_coef': 5e-3,
        'ortho_coef': 1.,
        'det1_coef': 1.,
        'use_cont_repr': True,

        'ip_avoid': False,
        'ip_bmodel': '/ps/project/common/moshpp/smpl/locked_head/neutral/model.pkl',
        'ip_max_collisions': 4,
        'ip_coef': 1e-2,

        'adam_beta1': 0.9,
        'best_model_fname': None,
        'log_every_epoch': 2,
        'expr_code': expr_code,
        'work_dir': work_dir,
        'num_epochs': 180,
        'dataset_dir': '/ps/project/smplbodyprior/BodyPrior/VPoser/data/0020_06_cmu_T3/smpl/pytorch/final_data',
    }

    vp_trainer = vposer_trainer(work_dir, ps=Configer(default_ps_fname=default_ps_fname, **params))
    ps = vp_trainer.ps

    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))

    expr_message = '[%s] %d H neurons, latentD=%d, batch_size=%d, beta=%.1e,  kl_coef = %.1e\n' \
                   % (ps.expr_code, ps.num_neurons, ps.latentD, ps.batch_size, ps.adam_beta1, ps.kl_coef)
    expr_message += 'Using [On the Continuity of Rotation Representations in Neural Networks]\n'
    expr_message += 'removing hand joints to be compatible with SMPLH\n'
    expr_message += 'lower KL\n'
    expr_message += 'Trained on CMU with faster dataloader\n'
    expr_message += '\n'

    vp_trainer.logger(expr_message)
    vp_trainer.perform_training()
    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))