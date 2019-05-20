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
import glob

class AMASSDataset(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)
    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data['pose_aa'] = data['pose_aa'].view(1,52,3)[:,1:22]
        data['pose_matrot'] = data['pose_matrot'].view(1,52,9)[:,1:22]
        return data

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    import trimesh

    batch_size = 10
    ds_dir = '/ps/project/humanbodyprior/BodyPrior/VPoser/data/20190313_cmu_T3/smpl/pytorch/final_data/vald'
    ds = AMASSDataset(dataset_dir=ds_dir)
    print(len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

    model_pklpath = '/ps/project/common/moshpp/smpl/locked_head/female/model.npz'
    bm = BodyModel(model_pklpath, model_type='smpl', batch_size=batch_size)

    for i_batch, sample_batched in enumerate(dataloader):

        vertices = c2c(bm.forward(pose_body=sample_batched['pose_aa'], betas=sample_batched['betas'][:,:10]).v)[1]
        faces = c2c(bm.f)

        mesh = trimesh.base.Trimesh(vertices, faces).show()

