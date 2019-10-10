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
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.12.13

import numpy as np

import torch
import torch.nn as nn

from smplx.lbs import lbs
# from human_body_prior.body_model.lbs import lbs
import os


class BodyModel(nn.Module):

    def __init__(self,
                 bm_path,
                 params=None,
                 num_betas=10,
                 batch_size=1,
                 num_dmpls=None, path_dmpl=None,
                 num_expressions=10,
                 use_posedirs=True,
                 dtype=torch.float32):

        super(BodyModel, self).__init__()

        '''
        :param bm_path: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
                if betas are provided in params, num_betas would be overloaded with number of thoes betas
        :param batch_size: number of smpl vertices to get
        :param device: default on gpu
        :param dtype: float precision of the compuations
        :return: verts, trans, pose, betas
        '''
        # Todo: if params the batchsize should be read from one of the params

        self.dtype = dtype

        if params is None: params = {}

        # -- Load SMPL params --
        if '.npz' in bm_path:
            smpl_dict = np.load(bm_path, encoding='latin1')
        else:
            raise ValueError('bm_path should be either a .pkl nor .npz file')

        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano.')

        self.use_dmpl = False
        if num_dmpls is not None:
            if path_dmpl is not None:
                self.use_dmpl = True
            else:
                raise (ValueError('path_dmpl should be provided when using dmpls!'))

        if self.use_dmpl and self.model_type in ['smplx', 'mano']: raise (
            NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))

        # Mean template vertices
        v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)
        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        self.register_buffer('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32))

        if len(params):
            if 'betas' in params.keys():
                num_betas = params['betas'].shape[1]
            if 'dmpls' in params.keys():
                num_dmpls = params['dmpls'].shape[1]

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas

        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=dtype))

        if self.model_type == 'smplx':
            begin_shape_id = 300 if smpl_dict['shapedirs'].shape[-1] > 300 else 10
            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            self.register_buffer('exprdirs', torch.tensor(exprdirs, dtype=dtype))

            expression = torch.tensor(np.zeros((batch_size, num_expressions)), dtype=dtype, requires_grad=True)
            self.register_parameter('expression', nn.Parameter(expression, requires_grad=True))

        if self.use_dmpl:
            dmpldirs = np.load(path_dmpl)['eigvec']

            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.register_buffer('dmpldirs', torch.tensor(dmpldirs, dtype=dtype))

        # Regressor for joint locations given shape - 6890 x 24
        self.register_buffer('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.register_buffer('posedirs', torch.tensor(posedirs, dtype=dtype))
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))

        if 'trans' in params.keys():
            trans = params['trans']
        else:
            trans = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        # if self.model_type in ['smpl', 'smplh']:
        root_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        # pose_body
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            pose_body = torch.tensor(np.zeros((batch_size, 63)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_body', nn.Parameter(pose_body, requires_grad=True))

        # pose_hand
        if 'pose_hand' in params.keys():
            pose_hand = params['pose_hand']
        else:
            if self.model_type in ['smpl']:
                pose_hand = torch.tensor(np.zeros((batch_size, 1 * 3 * 2)), dtype=dtype, requires_grad=True)
            elif self.model_type in ['smplh', 'smplx']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3 * 2)), dtype=dtype, requires_grad=True)
            elif self.model_type in ['mano']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('pose_hand', nn.Parameter(pose_hand, requires_grad=True))

        # face poses
        if self.model_type == 'smplx':
            pose_jaw = torch.tensor(np.zeros((batch_size, 1 * 3)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_jaw', nn.Parameter(pose_jaw, requires_grad=True))
            pose_eye = torch.tensor(np.zeros((batch_size, 2 * 3)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_eye', nn.Parameter(pose_eye, requires_grad=True))

        if 'betas' in params.keys():
            betas = params['betas']
        else:
            betas = torch.tensor(np.zeros((batch_size, num_betas)), dtype=dtype, requires_grad=True)
        self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))

        if self.use_dmpl:
            if 'dmpls' in params.keys():
                dmpls = params['dmpls']
            else:
                dmpls = torch.tensor(np.zeros((batch_size, num_dmpls)), dtype=dtype, requires_grad=True)
            self.register_parameter('dmpls', nn.Parameter(dmpls, requires_grad=True))
        self.batch_size = batch_size

    def r(self):
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        return c2c(self.forward().v)

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, v_template =None, **kwargs):
        '''

        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        '''
        assert not (v_template  is not None and betas  is not None), ValueError('vtemplate and betas could not be used jointly.')
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:  root_orient = self.root_orient
        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:  pose_body = self.pose_body
            if pose_hand is None:  pose_hand = self.pose_hand
        elif self.model_type == 'smplx':
            if pose_body is None:  pose_body = self.pose_body
            if pose_hand is None:  pose_hand = self.pose_hand
            if pose_jaw is None:  pose_jaw = self.pose_jaw
            if pose_eye is None:  pose_eye = self.pose_eye
        elif self.model_type in ['mano', 'mano']:
            if pose_hand is None:  pose_hand = self.pose_hand
        
        if pose_hand is None:  pose_hand = self.pose_hand

        if trans is None: trans = self.trans
        if v_template is None: v_template = self.v_template
        if betas is None: betas = self.betas

        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=1)
        elif self.model_type == 'smplx':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand],
                                  dim=1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        elif self.model_type in ['mano', 'mano']:
            full_pose = torch.cat([root_orient, pose_hand], dim=1)

        if self.use_dmpl:
            if dmpls is None: dmpls = self.dmpls
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == 'smplx':
            if expression is None: expression = self.expression
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                            shapedirs=shapedirs, posedirs=self.posedirs,
                            J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
                            lbs_weights=self.weights,
                            dtype=self.dtype)

        Jtr = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['f'] = self.f
        res['betas'] = self.betas
        res['Jtr'] = Jtr  # Todo: ik can be made with vposer

        if self.model_type == 'smpl':
            res['pose_body'] = pose_body
        elif self.model_type == 'smplh':
            res['pose_body'] = pose_body
            res['pose_hand'] = pose_hand
        elif self.model_type == 'smplx':
            res['pose_body'] = pose_body
            res['pose_hand'] = pose_hand
            res['pose_jaw'] = pose_jaw
            res['pose_eye'] = pose_eye
        elif self.model_type in ['mano', 'mano']:
            res['pose_hand'] = pose_hand
        res['full_pose'] = full_pose

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res


class BodyModelWithPoser(BodyModel):
    def __init__(self, poser_type='vposer', smpl_exp_dir='0020_06', mano_exp_dir=None, **kwargs):
        '''
        :param poser_type: vposer/gposer
        :param kwargs:
        '''
        super(BodyModelWithPoser, self).__init__(**kwargs)
        self.poser_type = poser_type

        self.use_hands = False if mano_exp_dir is None else True

        if self.poser_type == 'vposer':

            if self.model_type == 'smpl':
                from human_body_prior.tools.model_loader import load_vposer as poser_loader

                self.poser_body_pt, self.poser_body_ps = poser_loader(smpl_exp_dir)
                self.poser_body_pt.to(self.trans.device)

                poZ_body = torch.tensor(np.zeros([self.batch_size, self.poser_body_ps.latentD]), requires_grad=True,
                                        dtype=self.trans.dtype)
                self.register_parameter('poZ_body', nn.Parameter(poZ_body, requires_grad=True))
                self.pose_body.requires_grad = False

            elif self.model_type in ['smplh', 'smplx']:
                # from experiments.nima.body_prior.tools_pt.load_vposer import load_vposer as poser_loader

                from human_body_prior.tools.model_loader import load_vposer as poser_loader
                # body
                self.poser_body_pt, self.poser_body_ps = poser_loader(smpl_exp_dir)
                self.poser_body_pt.to(self.trans.device)

                poZ_body = self.pose_body.new(np.zeros([self.batch_size, self.poser_body_ps.latentD]))
                self.register_parameter('poZ_body', nn.Parameter(poZ_body, requires_grad=True))
                self.pose_body.requires_grad = False

                if self.use_hands:
                    # hand left
                    self.poser_handL_pt, self.poser_handL_ps = poser_loader(mano_exp_dir)
                    self.poser_handL_pt.to(self.trans.device)

                    poZ_handL = self.pose_hand.new(np.zeros([self.batch_size, self.poser_handL_ps.latentD]))
                    self.register_parameter('poZ_handL', nn.Parameter(poZ_handL, requires_grad=True))

                    # hand right
                    self.poser_handR_pt, self.poser_handR_ps = poser_loader(mano_exp_dir)
                    self.poser_handR_pt.to(self.trans.device)

                    poZ_handR = self.pose_hand.new(np.zeros([self.batch_size, self.poser_handR_ps.latentD]))
                    self.register_parameter('poZ_handR', nn.Parameter(poZ_handR, requires_grad=True))
                    self.pose_hand.requires_grad = False

            elif self.model_type in ['mano_left', 'mano_right']:
                if not self.use_hands: raise ('When using MANO only VPoser you have to provide mano_exp_dir')

                from human_body_prior.tools.model_loader import load_vposer as poser_loader

                self.poser_hand_pt, self.poser_hand_ps = poser_loader(mano_exp_dir)
                self.poser_hand_pt.to(self.trans.device)

                poZ_hand = self.pose_hand.new(np.zeros([self.batch_size, self.poser_hand_ps.latentD]))
                self.register_parameter('poZ_hand', nn.Parameter(poZ_hand, requires_grad=True))
                self.pose_hand.requires_grad = False

    def forward(self, poZ_body=None, **kwargs):
        if poZ_body is None:  poZ_body = self.poZ_body

        if self.poser_type == 'vposer':
            if self.model_type in ['smpl', 'smplh', 'smplx']:
                pose = self.poser_body_pt.decode(poZ_body, output_type='aa').view(self.batch_size, -1)

                if pose.shape[1] > 63:
                    pose_body = pose[:, 3:66]
                    root_orient = pose[:, :3]
                else:
                    pose_body = pose[:, :63]
                    root_orient = None

                if self.use_hands and self.model_type in['smplh', 'smplx']:
                    pose_handL = self.poser_handL_pt.decode(self.poZ_handL, output_type='aa').view(self.batch_size, -1)
                    pose_handR = self.poser_handR_pt.decode(self.poZ_handR, output_type='aa').view(self.batch_size, -1)
                    pose_hand = torch.cat([pose_handL, pose_handR], dim=1)
                else:
                    pose_hand = None

                new_body = super(BodyModelWithPoser, self).forward(pose_body=pose_body, root_orient=root_orient, pose_hand=pose_hand, **kwargs)
                new_body.poZ_body = poZ_body


            if self.model_type in ['mano_left', 'mano_right']:
                pose_hand = self.poser_hand_pt.decode(self.poZ_hand, output_type='aa').view(self.batch_size, -1)
                # new_body = BodyModel.forward(self, pose_hand=pose_hand)
                new_body = super(BodyModelWithPoser, self).forward(pose_hand=pose_hand, **kwargs)

        else:
            new_body = BodyModel.forward(self)

        return new_body

    def randomize_pose(self):
        if self.poser_type == 'vposer':
            if self.model_type in ['smpl', 'smplh', 'smplx']:
                with torch.no_grad():
                    poZ_body = self.poZ_body.new(np.random.randn(*list(self.poZ_body.shape))).detach()
                    pose = self.poser_body_pt.decode(poZ_body, output_type='aa').view(self.batch_size, -1)
                    if pose.shape[1] > 63:
                        pose_body = pose[:,3:66]
                        root_orient = pose[:,:3]
                    else:
                        pose_body = pose[:,:63]
                        root_orient = None

                    self.pose_body.data[:] = pose_body
                    self.root_orient.data[:] = root_orient

            if self.use_hands and self.model_type in ['smplh', 'smplx']:
                with torch.no_grad():

                    self.poZ_handL.data[:] = self.poZ_handL.new(np.random.randn(*list(self.poZ_handL.shape))).detach()
                    self.poZ_handR.data[:] = self.poZ_handR.new(np.random.randn(*list(self.poZ_handR.shape))).detach()

                        pose_handL = self.poser_handL_pt.decode(self.poZ_handL, output_type='aa').view(self.batch_size, -1)
                        pose_handR = self.poser_handR_pt.decode(self.poZ_handR, output_type='aa').view(self.batch_size, -1)
                        self.pose_hand.data[:] = torch.cat([pose_handL, pose_handR], dim=1)

    def untagnle_interpenetrations(self, max_collisions=8, sigma=1e-3):
        bmip = BodyInterpenetration(self, max_collisions=max_collisions, sigma=sigma)
        old_body_v = self.forward().v.detach()
        old_poses = [var[1].detach() for var in self.named_parameters() if 'poz' in var[0].lower()]
        def compute_loss(new_body, new_poses):
            pose_wt = 1e-6
            ip_wt = 100.
            data_wt = 1.e5

            data_loss = data_wt * torch.pow(old_body_v - new_body.v,2).mean(dim=0).sum()
            pose_loss = pose_wt * torch.cat([torch.pow(pose,2) for pose in new_poses],1).mean(dim=0).sum()
            ip_loss =  ip_wt * torch.pow(bmip(new_body), 2).mean(dim=0).sum()
            total_loss = ip_loss + pose_loss + data_loss
            return total_loss

        def closure(free_vars):
            new_body = self.forward()
            loss_total = compute_loss(new_body, new_poses=free_vars)
            loss_total.backward()
            return loss_total

        def perform(max_iter=300, ftol=1e-4, gtol=1e-3):
            cur_ip_loss = bmip(self.forward()).mean().item()
            if not cur_ip_loss>2.0:
                print('No need for untangling')
                return cur_ip_loss
            else: print(cur_ip_loss)

            free_vars = [var[1] for var in self.named_parameters() if 'poz' in var[0].lower()]
            # from torch import optim
            # optimizer = optim.Adam(free_vars, lr=1e-2)
            # optimizer.zero_grad()
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(max_iter // 3), gamma=0.1)
            from human_body_prior.optimizers.lbfgs_ls import LBFGS
            # from torch.optim.lbfgs import LBFGS # will adopt this shortly from pytorch 1.2.0
            optimizer = LBFGS(free_vars, lr=1e-8, max_iter=max_iter, line_search_fn='strong_Wolfe', tolerance_grad=1e-2, tolerance_change=1e-4)
            optimizer.zero_grad()
            prev_loss = None
            # for opt_it in range(max_iter):
            #     scheduler.step()
            #     loss = optimizer.step(lambda : closure(free_vars))
            #
            #     if opt_it > 0 and prev_loss is not None and ftol > 0:
            #         loss_rel_change = rel_change(prev_loss, loss.item())
            #
            #         if loss_rel_change <= ftol:
            #             break
            #
            #     if all([torch.abs(var.grad.view(-1).max()).item() < gtol for var in free_vars if var.grad is not None]):
            #         break
            #
            #     prev_loss = loss.item()
            # print max_iter
            prev_loss = optimizer.step(lambda : closure(free_vars))
            return bmip(self.forward()).mean().item()

        return perform

class BodyInterpenetration(nn.Module):
    def __init__(self, bm, max_collisions=8, sigma=1e-3, filter_faces=True):
        super(BodyInterpenetration, self).__init__()
        self.bm = bm
        self.model_type = bm.model_type
        nv = bm.shapedirs.shape[0]
        device = bm.f.device
        if 'cuda' not in str(device): raise NotImplementedError('Interpenetration term is only avaialble for body models on GPU.')
        try:import mesh_intersection
        except:raise('Optional package mesh_intersection is required for this functionality. Please install from https://github.com/vchoutas/torch-mesh-isect.')
        from mesh_intersection.bvh_search_tree import BVH
        from mesh_intersection.loss import DistanceFieldPenetrationLoss
        self.search_tree = BVH(max_collisions=max_collisions)
        self.pen_distance = DistanceFieldPenetrationLoss(
            sigma=sigma, point2plane=False,
            vectorized=True, penalize_outside=True)

        self.filter_faces = None
        if filter_faces:
            if self.model_type == 'mano':
                import sys
                sys.stderr.write('Filter faces is not available for MANO model yet!')
            else:
                #import cPickle as pickle
                import pickle
                from mesh_intersection.filter_faces import FilterFaces
                # ign_part_pairs: The pairs of parts where collisions will be ignored
                # here 1: LeftTigh, 2: RightTigh, 6:Spine1, 9:Spine2, 12:Neck, 15:Head, 16:LeftUpperArm, 17:RightUpperArm, 22:Jaw
                ign_part_pairs = ["9,16", "9,17", "6,16", "6,17", "1,2"] + (["12,15"] if self.model_type in ['smpl', 'smplh'] else ["12,22"])

                part_segm_fname = os.path.join(os.path.dirname(__file__),'parts_segm/%s/parts_segm.pkl'%('smplh' if self.model_type in ['smpl', 'smplh'] else self.model_type))

                with open(part_segm_fname, 'rb') as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file, encoding='latin1')

                faces_segm = face_segm_data['segm']
                faces_parents = face_segm_data['parents']
                # Create the module used to filter invalid collision pairs
                self.filter_faces = FilterFaces(
                    faces_segm=faces_segm, faces_parents=faces_parents,
                    ign_part_pairs=ign_part_pairs).to(device=device)

        batched_f = bm.f.clone().unsqueeze(0).repeat([bm.batch_size, 1, 1]).type(torch.long)
        self.faces_ids = batched_f + (torch.arange(bm.batch_size, dtype=torch.long).to(device) * nv)[:, None, None]

    def forward(self, bm_forwarded = None, **kwargs):
        if bm_forwarded is None: bm_forwarded = self.bm.forward(**kwargs)
        # triangles = new_body.v[self.batched_f]
        triangles = bm_forwarded.v.view([-1, 3])[self.faces_ids]
        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)

        if self.filter_faces is not None: collision_idxs = self.filter_faces(collision_idxs)
        pen_loss = self.pen_distance(triangles, collision_idxs)
        return pen_loss