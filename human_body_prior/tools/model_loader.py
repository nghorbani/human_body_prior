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
# Code Developed by: Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# 2018.01.02

import os
import numpy as np


def expid2model(expr_dir, model_type):
    from configer import Configer
    import os, glob

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    trained_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    try_num = os.path.basename(trained_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % trained_model_fname))

    default_ps_fname = os.path.join(expr_dir, '%s_vposer_%s_settings.ini' % (try_num, model_type.replace('_left', '').replace('_right', '')))
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir)

    return ps, trained_model_fname

def load_vposer(expr_dir, model_type='smpl', use_snapshot_model = False):
    '''

    :param expr_dir:
    :param model_type: mano/smpl
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch

    ps, trained_model_fname = expid2model(expr_dir, model_type=model_type)
    if use_snapshot_model:

        vposer_path = os.path.join(expr_dir, 'vposer_%s_pt.py'%model_type.replace('_left','').replace('_right',''))

        spec = importlib.util.spec_from_file_location('VPoser', vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        vposer_pt = getattr(module, 'VPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        if model_type == 'smpl':
            from human_body_prior.train.train_vposer_smpl import VPoser
        else:
            raise NotImplementedError
        vposer_pt = VPoser(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    vposer_pt.eval()

    return vposer_pt, ps


def extract_weights_asnumpy(exp_id, model_type='smpl', use_snapshot_model= False):
    from human_body_prior.tools.omni_tools import makepath
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    vposer_pt, vposer_ps = load_vposer(exp_id, model_type=model_type, use_snapshot_model=use_snapshot_model)

    save_wt_dir = makepath(os.path.join(vposer_ps.work_dir, 'weights_npy'))

    weights = {}
    for var_name, var in vposer_pt.named_parameters():
        weights[var_name] = c2c(var)
    np.savez(os.path.join(save_wt_dir,'vposerWeights.npz'), **weights)

    print(('Dumped weights as numpy arrays to %s'%save_wt_dir))
    return vposer_ps, weights

if __name__ == '__main__':
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    expr_dir = '/ps/project/humanbodyprior/BodyPrior/VPoser/smpl/pytorch/0020_06_amass'
    vposer_pt, ps = load_vposer(expr_dir, model_type='smpl', use_snapshot_model=False)
    pose = c2c(vposer_pt.sample_poses(10, seed=100)[0,0])
    print(pose.shape)
    print(pose[:])