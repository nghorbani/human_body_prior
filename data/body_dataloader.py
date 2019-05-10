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
#
# 2018.01.02

import os
import torch
from torch.utils.data import Dataset
import torchgeometry as tgm

class AMASSDataset(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, output_type='matrot', dtype=torch.float32):
        # Todo add gender
        self.ds = {'pose':torch.load(os.path.join(dataset_dir, 'data_pose.pt')),
                   'shape': torch.load(os.path.join(dataset_dir, 'data_shape.pt')),
                   }
        self.dtype = dtype
        self.output_type = output_type

    def __len__(self):
       return len(self.ds['pose'])

    def __getitem__(self, idx):
        return self.fetch_data(idx, self.output_type)

    def fetch_data(self, idx, output_type='matrot'):
        pose = self.ds['pose'][idx]
        if output_type == 'matrot':
            pose = tgm.angle_axis_to_rotation_matrix(pose.view(-1, 3))[:, :3, :3].contiguous().view(1, -1, 9)
        else:
            pose = pose.view(1, -1, 3)

        sample = {'pose': pose.type(self.dtype),
                  'shape': self.ds['shape'][idx].type(self.dtype),
                  }
        return sample

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from tools.body_model import BodyModel
    from tools.omni_tools import copy2cpu as c2c
    import trimesh

    batch_size = 10
    ds_dir = '/ps/project/smplbodyprior/BodyPrior/VPoser/data/20190313_cmu/smpl/pytorch/vald'
    ds = AMASSDataset(dataset_dir=ds_dir, dtype=torch.float32, output_type='aa')
    print(len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

    model_pklpath = '/ps/project/common/moshpp/smpl/locked_head/female/model.npz'
    bm = BodyModel(model_pklpath, model_type='smpl', batch_size=batch_size)

    for i_batch, sample_batched in enumerate(dataloader):

        vertices = c2c(bm.forward(pose_body=sample_batched['pose'][:,0,1:22].view(-1,63), betas=sample_batched['shape'][:,:10]).v)[1]
        faces = c2c(bm.f)

        mesh = trimesh.base.Trimesh(vertices, faces).show()

