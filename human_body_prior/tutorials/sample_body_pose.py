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
# 2019.05.10

__all__ = ['sample_vposer']

import numpy as np

import os

def dump_vposer_samples(pose_body, out_imgpath, save_obj=True):
    from human_body_prior.tools.viz_tools import vis_smpl_params, imagearray2file, smpl_params2ply
    from human_body_prior.tools.omni_tools import makepath
    from human_body_prior.body_model.body_model import BodyModel
    bm_path = '/ps/project/common/moshpp/smpl/locked_head/male/model.npz'
    bm = BodyModel(bm_path, 'smpl')

    imgs = vis_smpl_params(bm, pose_body=pose_body)
    imagearray2file(imgs.reshape(1, -1, 1, 400, 400, 3), out_imgpath)

    np.savez(out_imgpath.replace('.png', '.npz'), pose=pose_body)

    if save_obj:
        im_id = os.path.basename(out_imgpath).split('.')[0]
        out_dir = makepath(os.path.join(os.path.dirname(out_imgpath), '%s_ply'%im_id))
        smpl_params2ply(bm, out_dir=out_dir, pose_body=pose_body)

    print('Saved image: %s' % out_imgpath)

    return True

def sample_vposer(exp_dir, num_samples=5):
    from human_body_prior.tools.omni_tools import id_generator, makepath
    from human_body_prior.tools.vposer_loader_pt import vposer_loader_pt
    from human_body_prior.tools.omni_tools import copy2cpu

    vposer_pt, ps = vposer_loader_pt(exp_dir, model_type='smpl')

    sampled_pose_body = copy2cpu(vposer_pt.sample_poses(num_poses=num_samples))

    out_dir = makepath(os.path.join(ps.work_dir, 'evaluations', 'pose_generation'))
    out_imgpath = os.path.join(out_dir, '%s.png' % id_generator(6))

    dump_vposer_samples(sampled_pose_body, out_imgpath)
    print('Dumped samples at %s'%out_dir)
    return sampled_pose_body


if __name__ == '__main__':

    expr_dir = '../expriments/VPoser/smpl/pytorch/0020_06_amass'
    sample_vposer(expr_dir, 5, save_samples=True)