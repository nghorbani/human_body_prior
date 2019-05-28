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
from human_body_prior.tools.omni_tools import makepath, id_generator, log2file
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.data.dataloader import VPoserDS

from torch.utils.data import DataLoader


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
    def __init__(self, num_neurons, latentD, data_shape, use_cont_repr=True):
        super(VPoser, self).__init__()

        self.latentD = latentD
        self.use_cont_repr = use_cont_repr

        n_features = np.prod(data_shape)
        self.num_joints = data_shape[1]

        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=.1, inplace=False)

        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_out = nn.Linear(num_neurons, self.num_joints* 6)

    def encode(self, Pin):
        '''

        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        Xout = Pin.view(Pin.size(0), -1)  # flatten input
        Xout = self.bodyprior_enc_bn1(Xout)

        Xout = F.leaky_relu(self.bodyprior_enc_fc1(Xout), negative_slope=.2)
        Xout = self.bodyprior_enc_bn2(Xout)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc2(Xout), negative_slope=.2)
        return torch.distributions.normal.Normal(self.bodyprior_enc_mu(Xout), F.softplus(self.bodyprior_enc_logvar(Xout)))

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

        Xout = Xout.view([-1, 1, self.num_joints, 9])
        if output_type == 'aa': return VPoser.matrot2aa(Xout)
        return Xout

    def forward(self, Pin, input_type='matrot', output_type='matrot'):
        '''

        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert output_type in ['matrot', 'aa']
        # if input_type == 'aa': Pin = VPoser.aa2matrot(Pin)
        q_z = self.encode(Pin)
        q_z_sample = q_z.rsample()
        Prec = self.decode(q_z_sample)
        if output_type == 'aa': Prec = VPoser.matrot2aa(Prec)

        #return Prec, q_z.mean, q_z.sigma
        return {'pose':Prec, 'mean':q_z.mean, 'std':q_z.scale}

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
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        return pose_body_matrot


class VPoserTrainer:

    def __init__(self, work_dir, ps):
        from tensorboardX import SummaryWriter

        from human_body_prior.tools.omni_tools import log2file, makepath

        self.pt_dtype = torch.float64 if ps.fp_precision == '64' else torch.float32

        torch.manual_seed(ps.seed)

        starttime = datetime.now().replace(microsecond=0)
        ps.work_dir = makepath(work_dir, isfile=False)

        logger = log2file(os.path.join(work_dir, '%s.log' % ps.expr_code))

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        shutil.copy2(sys.argv[0], work_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda: torch.cuda.empty_cache()
        self.comp_device = torch.device("cuda:%d"%ps.cuda_id if torch.cuda.is_available() else "cpu")

        logger('%d CUDAs available!' % torch.cuda.device_count())

        gpu_brand= torch.cuda.get_device_name(ps.cuda_id) if use_cuda else None
        logger('Training with %s [%s]' % (self.comp_device,gpu_brand)  if use_cuda else 'Training on CPU!!!')
        logger('Base dataset_dir is %s'%ps.dataset_dir)

        # if ps.ip_avoid:
        #     from tools_torch.smpl_pt import BodyInterpenetration
        #     from nima.tools_torch.smpl_pt import BodyModel
        #     bm = BodyModel(model_pklpath=ps.ip_bmodel, model_type=ps.model_type, batch_size=ps.batch_size).to('cuda')
        #     self.body_interpenetration = BodyInterpenetration(bm)

        kwargs = {'num_workers': ps.n_workers}
        # kwargs = {'num_workers': ps.n_workers, 'pin_memory': True} if use_cuda else {'num_workers': ps.n_workers}
        ds_train = VPoserDS(dataset_dir=os.path.join(ps.dataset_dir, 'train'))
        self.ds_train = DataLoader(ds_train, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_val = VPoserDS(dataset_dir=os.path.join(ps.dataset_dir, 'vald'))
        self.ds_val = DataLoader(ds_val, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_test = VPoserDS(dataset_dir=os.path.join(ps.dataset_dir, 'test'))
        self.ds_test = DataLoader(ds_test, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        logger('Train dataset size %.2f M' % (len(self.ds_train.dataset)*1e-6))
        logger('Validation dataset size %d' % len(self.ds_val.dataset))
        logger('Test dataset size %d' % len(self.ds_test.dataset))

        ps.data_shape = list(ds_val[0]['pose'].shape)
        self.vposer_model = VPoser(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape,
                                   use_cont_repr=ps.use_cont_repr)#.to(self.comp_device)

        self.vposer_model = nn.DataParallel(self.vposer_model)
        self.vposer_model.to(self.comp_device)

        # enc_varlist = [var[1] for var in self.vposer_model.named_parameters() if 'bodyprior_enc' in var[0]]
        # dec_varlist = [var[1] for var in self.vposer_model.named_parameters() if 'bodyprior_dec' in var[0]]
        varlist = [var[1] for var in self.vposer_model.named_parameters()]

        params_count = sum(p.numel() for p in varlist if p.requires_grad)
        # enc_params_count = sum(p.numel() for p in enc_varlist if p.requires_grad)
        # dec_params_count = sum(p.numel() for p in dec_varlist if p.requires_grad)
        # logger('Encoder Trainable Parameters Count %2.2f M and in Decoder: %2.2f M.' % (
        # enc_params_count * 1e-6, dec_params_count * 1e-6,))
        # logger('Total Trainable Parameters Count is %2.2f M.' % ((dec_params_count + enc_params_count) * 1e-6))
        logger('Total Trainable Parameters Count is %2.2f M.' % ((params_count) * 1e-6))

        self.optimizer = optim.Adam(varlist, betas=(ps.adam_beta1, 0.999), lr=ps.base_lr, weight_decay=ps.reg_coef)

        self.logger = logger
        self.best_loss_total = np.inf
        self.try_num = ps.try_num
        self.epochs_completed = 0
        self.ps = ps

        if ps.best_model_fname is not None:
            self.vposer_model.module.load_state_dict(torch.load(ps.best_model_fname, map_location=self.comp_device))
            logger('Restored model from %s'%ps.best_model_fname)

        chose_ids = np.random.choice(list(range(len(ds_val))), size=ps.num_bodies_to_display, replace=False, p=None)
        data_all = {}
        for id in chose_ids:
            for k, v in ds_val[id].items():
                if k in data_all.keys():
                    data_all[k] = torch.cat([data_all[k], v[np.newaxis]], dim=0)
                else:
                    data_all[k] = v[np.newaxis]

        self.vis_dorig = {k: data_all[k].to(self.comp_device) for k in data_all.keys()}

        self.bm_path = '/ps/project/common/moshpp/smplh/locked_head/neutral/model.npz'
        self.bm = BodyModel(self.bm_path, 'smplh', batch_size=self.ps.batch_size, use_posedirs=True).to(self.comp_device)

        # self.swriter.add_graph(self.vposer_model, self.vis_dorig, True)

    def train(self):
        self.vposer_model.train()
        save_every_it = len(self.ds_train) / self.ps.log_every_epoch
        train_loss_dict = {}
        for it, data in enumerate(self.ds_train):

            dorig = data['pose'].to(self.comp_device)
            self.optimizer.zero_grad()
            drec = self.vposer_model(dorig, input_type='aa', output_type='aa')
            loss_total, cur_loss_dict = self.compute_loss(dorig, drec)
            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = VPoserTrainer.creat_loss_message(cur_train_loss_dict, expr_code=self.ps.expr_code,
                                                             epoch_num=self.epochs_completed, it=it,
                                                             try_num=self.try_num, mode='train')

                self.logger(train_msg)
                self.swriter.add_histogram('q_z_sample', c2c(drec['mean']), it)

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self, split_name= 'vald'):
        self.vposer_model.eval()
        eval_loss_dict = {}
        data = self.ds_val if split_name == 'vald' else self.ds_test
        with torch.no_grad():
            for data in data:
                dorig = data['pose'].to(self.comp_device)
                drec = self.vposer_model(dorig, input_type='aa', output_type='aa')
                _, cur_loss_dict = self.compute_loss(dorig, drec)
                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, dorig, drec):
        prec = drec['pose']
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        batch_size = dorig.size(0)
        device = dorig.device
        dtype = dorig.dtype
        latentD = q_z.mean.size(1)
        MESH_SCALER = 1000

        #loss_rec = (1. - self.ps.kl_coef) * torch.mean(torch.sum(torch.pow(dorig - prec, 2), dim=[1, 2, 3]))
        mesh_orig = self.bm(pose_body=dorig.view(self.ps.batch_size,-1)).v*MESH_SCALER
        mesh_rec = self.bm(pose_body=prec.view(self.ps.batch_size,-1)).v*MESH_SCALER
        loss_mesh_rec =  (1. - self.ps.kl_coef) * torch.mean(torch.abs(mesh_orig - mesh_rec))

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
                     'loss_mesh_rec': loss_mesh_rec,
                     # 'loss_ortho':loss_ortho,
                     # 'loss_det1':loss_det1,
                     }

        if self.vposer_model.training and self.epochs_completed < 10:
            loss_dict['loss_pose_rec'] = (1. - self.ps.kl_coef) * torch.mean(torch.sum(torch.pow(dorig - prec, 2), dim=[1, 2, 3]))

        if self.ps.ip_avoid:
            pose = VPoser.matrot2aa(prec).view(batch_size, -1)
            bm_forwarded = self.body_interpenetration.bm(pose_body=pose)
            loss_dict['loss_ip_rec'] = self.ps.ip_coef * torch.mean(self.body_interpenetration(bm_forwarded=bm_forwarded))

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

        vis_bm =  BodyModel(self.bm_path, 'smplh', num_betas=16).to(self.comp_device)
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
                eval_msg = VPoserTrainer.creat_loss_message(eval_loss_dict, expr_code=self.ps.expr_code,
                                                            epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                            try_num=self.try_num, mode='evald')
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.ps.best_model_fname = makepath(os.path.join(self.ps.work_dir, 'snapshots', 'TR%02d_E%03d.pt' % (
                    self.try_num, self.epochs_completed)), isfile=True)
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.vposer_model.module.state_dict() if isinstance(self.vposer_model, torch.nn.DataParallel) else self.vposer_model.state_dict(), self.ps.best_model_fname)

                    imgname = '[%s]_TR%02d_E%03d.png' % (self.ps.expr_code, self.try_num, self.epochs_completed)
                    imgpath = os.path.join(self.ps.work_dir, 'images', imgname)
                    # VPoserTrainer.vis_results(self.vis_dorig, self.vposer_model, bm=vis_bm, imgpath=imgpath)

                else:
                    self.logger(eval_msg)

                # prec, _ = self.vposer_model(self.vis_dorig)
                # R = prec.view([-1, 23, 3, 3])
                # det_R = torch.transpose(torch.stack([determinant_3d(R[:, jIdx, ...]) for jIdx in range(23)]), 0, 1)
                # self.swriter.add_histogram('det_R', c2c(det_R), epoch_num)

                self.swriter.add_scalars('train_loss/scalars', train_loss_dict, self.epochs_completed)
                self.swriter.add_scalars('eval_loss/scalars', eval_loss_dict, self.epochs_completed)
                self.swriter.add_scalars('total_loss/scalars', {'train_loss_total': train_loss_dict['loss_total'],
                                                                'evald_loss_total': eval_loss_dict['loss_total'], },
                                         self.epochs_completed)

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        self.logger('Best model path: %s\n' % self.ps.best_model_fname)

    @staticmethod
    def creat_loss_message(loss_dict, expr_code='XX', epoch_num=0, it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s: [T:%.2e] - [%s]' % (
        expr_code, try_num, epoch_num, it, mode, loss_dict['loss_total'], ext_msg)

    @staticmethod
    def vis_results(dorig, vposer_model, bm, imgpath):
        from human_body_prior.mesh import MeshViewer
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        import trimesh
        from human_body_prior.tools.omni_tools import colors
        from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

        from human_body_prior.tools.visualization_tools import imagearray2file
        from human_body_prior.train.vposer_smpl import VPoser

        view_angles = [0, 180, 90, -90]
        imw, imh = 800, 800
        batch_size = len(dorig['pose'])
        num_joints = dorig['pose'].shape[2]

        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.render_wireframe = True

        dorig_aa = dorig['pose'].view([-1,1, num_joints, 3])

        prec_aa = vposer_model(dorig_aa, input_type='aa', output_type='aa')['pose']
        prec_aa = prec_aa.view(batch_size,-1)
        if hasattr(vposer_model, 'module'):
            pgen_aa = vposer_model.module.sample_poses(num_poses=batch_size, output_type='aa')
        else:
            pgen_aa = vposer_model.sample_poses(num_poses=batch_size, output_type='aa')

        pgen_aa = pgen_aa.view(batch_size,-1)
        dorig_aa = dorig_aa.view(batch_size, -1)

        images = np.zeros([len(view_angles), batch_size, 1, imw, imh, 3])
        images_gen = np.zeros([len(view_angles), batch_size, 1, imw, imh, 3])
        for cId in range(0, batch_size):

            bm.pose_body.data[:] = bm.pose_body.new(dorig_aa[cId])
            orig_body_mesh = trimesh.Trimesh(vertices=c2c(bm().v[0]), faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))

            bm.pose_body.data[:] = bm.pose_body.new(prec_aa[cId])
            rec_body_mesh = trimesh.Trimesh(vertices=c2c(bm().v[0]), faces=c2c(bm.f), vertex_colors=np.tile(colors['blue'], (6890, 1)))

            bm.pose_body.data[:] = bm.pose_body.new(pgen_aa[cId])
            gen_body_mesh = trimesh.Trimesh(vertices=c2c(bm().v[0]), faces=c2c(bm.f), vertex_colors=np.tile(colors['blue'], (6890, 1)))

            all_meshes = [orig_body_mesh, rec_body_mesh, gen_body_mesh]
            # apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes([orig_body_mesh, rec_body_mesh], group_name='static')
                images[rId, cId, 0] = mv.render()
                mv.set_meshes([gen_body_mesh], group_name='static')
                images_gen[rId, cId, 0] = mv.render()

                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

        imagearray2file(images, imgpath)
        imagearray2file(images_gen, imgpath.replace('.png','_gen.png'))


def run_trainer(default_ps_fname):
    ps = Configer(default_ps_fname=default_ps_fname)
    vp_trainer = VPoserTrainer(ps.work_dir, ps)

    ps.dump_settings(os.path.join(ps.work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))

    expr_message = '\n[%s] %d H neurons, latentD=%d, batch_size=%d, beta=%.1e,  kl_coef = %.1e\n' \
                   % (ps.expr_code, ps.num_neurons, ps.latentD, ps.batch_size, ps.adam_beta1, ps.kl_coef)
    expr_message += 'Trained on CMU with faster dataloader\n'
    expr_message += 'Using SMPL to produce meshes of the bodies\n'
    expr_message += 'Reconstruction loss is L1 on meshes\n'
    expr_message += 'Using Batch Normalization\n'
    expr_message += 'Running on the cluster, hence the large batch size. accommodate the base_lr\n'
    expr_message += '\n'

    vp_trainer.logger(expr_message)
    vp_trainer.perform_training()
    ps.dump_settings(os.path.join(ps.work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))

    vp_trainer.logger(expr_message)

    test_loss_dict = vp_trainer.evaluate(split_name='test')
    vp_trainer.logger('Final loss on test set is %s' % (' | '.join(['%s = %.2e' % (k, v) for k, v in test_loss_dict.items()])))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a vposer given settings',formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config_path', dest="config_path", type=str, help='path to ini file for Configer.')

    args = parser.parse_args()

    run_trainer(args.config_path)
