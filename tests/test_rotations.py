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

import unittest

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.train.vposer_smpl import VPoser

import numpy as np
import cv2
import torch


class TestRotationConversions(unittest.TestCase):

    def test_aa2matrot(self):
        aa = np.random.randn(10, 3)
        cv2_matrot = []
        for id in range(aa.shape[0]):
            cv2_matrot.append(cv2.Rodrigues(aa[id:id+1])[0])
        cv2_matrot = np.array(cv2_matrot).reshape(-1,9)

        vposer_matrot = c2c(VPoser.aa2matrot(torch.tensor(aa))).reshape(-1,9)
        self.assertAlmostEqual(np.square((vposer_matrot - cv2_matrot)).sum(), 0.0)

    def test_matrot2aa(self):
        np.random.seed(100)
        aa = np.random.randn(10, 3)
        matrot = c2c(VPoser.aa2matrot(torch.tensor(aa))).reshape(-1,9)

        cv2_aa = []
        for id in range(matrot.shape[0]):
            cv2_aa.append(cv2.Rodrigues(matrot[id].reshape(3,3))[0])
        cv2_aa = np.array(cv2_aa).reshape(-1,3)

        vposer_aa = c2c(VPoser.matrot2aa(torch.tensor(matrot))).reshape(-1,3)
        self.assertAlmostEqual(np.square((vposer_aa - cv2_aa)).sum(), 0.0)


if __name__ == '__main__':
    unittest.main()