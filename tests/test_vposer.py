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
# 2018.01.02

import unittest

from human_body_prior.train.vposer_smpl import VPoser
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from configer import Configer

import numpy as np

class TestDistances(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(100)

    def test_samples(self):
        ''' given the same network weights, the random pose generator must produce the same pose for a seed'''
        ps = Configer(default_ps_fname='../human_body_prior/train/vposer_smpl_defaults.ini')
        vposer = VPoser(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape = ps.data_shape)
        body_pose_rnd = vposer.sample_poses(num_poses=1, seed=100)

        body_pose_gt = np.load('samples/body_pose_rnd.npz')['data']
        self.assertAlmostEqual(np.square((c2c(body_pose_rnd) - body_pose_gt)).sum(), 0.0)

if __name__ == '__main__':
    unittest.main()