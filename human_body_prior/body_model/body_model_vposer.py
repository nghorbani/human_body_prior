import os

import numpy as np
import torch
from torch import nn as nn

from human_body_prior.body_model.body_model import BodyModel


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

        if self.poser_type == 'vposer':
            if self.model_type in ['smpl', 'smplh', 'smplx']:
                if poZ_body is None:  poZ_body = self.poZ_body

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
                new_body = super(BodyModelWithPoser, self).forward(pose_hand=pose_hand, **kwargs)

        else:
            new_body = BodyModel.forward(self)

        return new_body

    def untangle_interpenetrations(self, max_collisions=8, sigma=1e-3):
        bmip = BodyInterpenetration(self, max_collisions=max_collisions, sigma=sigma)

        # def compute_loss(new_body, poZ_body, old_body_v):
        #     poZ_wt = 1e-6
        #     ip_wt = 100.
        #     data_wt = 1.e5
        #
        #     data_loss = data_wt * torch.pow(old_body_v - new_body.v,2).mean(dim=0).sum()
        #     poZ_loss = poZ_wt * torch.pow(poZ_body,2).mean(dim=0).sum()
        #     ip_loss =  ip_wt * torch.pow(bmip(new_body), 2).mean(dim=0).sum()
        #     total_loss = ip_loss + poZ_loss + data_loss
        #     return total_loss

        # def closure(poZ_body, old_body_v):
        #     new_body = self.forward()
        #     loss_total = compute_loss(new_body, poZ_body=poZ_body, old_body_v=old_body_v)
        #     loss_total.backward()
        #     return loss_total

        def ip_fit(optimizer, static_vars, gstep=0):
            def fit(free_vars):
                print(fit.gstep)
                fit.gstep += 1
                optimizer.zero_grad()

                new_body = self.forward(**free_vars)

                poZ_wt = 1e-6
                ip_wt = 100.
                data_wt = 1.e5

                #data_loss = data_wt * torch.pow(static_vars['old_body_v'] - new_body.v, 2).mean(dim=0).sum()
                #poZ_loss = poZ_wt * torch.pow(free_vars['poZ_body'], 2).mean(dim=0).sum()
                ip_loss = ip_wt * torch.pow(bmip(new_body), 2).mean(dim=0).sum()
                total_loss = ip_loss# + poZ_loss + data_loss

                fit.final_loss = total_loss

                return total_loss

            fit.gstep = gstep
            fit.final_loss = 0.0
            return fit

        def perform(poZ_body, max_iter=300):
            from human_body_prior.tools.omni_tools import copy2cpu as c2c

            old_body = self.forward(poZ_body=poZ_body)

            free_vars = {
                'poZ_body': poZ_body,
            }

            static_vars = {
                'old_body_v': old_body.v.detach().clone(),
            }

            cur_ip_loss = bmip(old_body).mean().item()
            print('Initial interpentration loss: %.2f' % cur_ip_loss)

            if not cur_ip_loss>2.0:
                print('No need for untangling')
                return poZ_body

            from human_body_prior.optimizers.lbfgs_ls import LBFGS
            optimizer = LBFGS(list(free_vars.values()), lr=1e-1, max_iter=max_iter, line_search_fn='strong_Wolfe', tolerance_grad = 1e-10, tolerance_change=1e-10)

            closure = ip_fit(optimizer, static_vars, gstep=0)

            optimizer.step(lambda: closure(free_vars))
            gstep = closure.gstep

            final_loss = bmip(self.forward(poZ_body=free_vars['poZ_body'])).mean().item()
            print('Final interpentration loss: %.2f' % final_loss)
            return free_vars['poZ_body']

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