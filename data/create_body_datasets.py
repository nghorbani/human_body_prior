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
import numpy as np
from tools.omni_tools import makepath
from experiments.nima.tools.rotations import euler2em, em2euler
import shutil, sys

import tables as pytables
from datetime import datetime
from experiments.nima.tools.omni_tools import log2file
import torch

def remove_Zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def create_dataset_records(datasets, amass_dir, out_dir, split_name, rnd_seed = 100):

    assert split_name in ['train', 'vald', 'test']
    np.random.seed(rnd_seed)

    makepath(out_dir, isfile=False)

    keep_rate = 0.3 # 30 percent

    data_pose = []
    data_shape = []
    for ds_name in datasets:
        ds_h5path = os.path.join(amass_dir, '%s.h5'%ds_name)
        if os.path.exists(ds_h5path):
            with pytables.open_file(ds_h5path) as f:
                ds_data = f.get_node('//%s'%ds_name.lower())
                N = len(ds_data)
                data_ids = np.random.choice(list(range(N)), int(keep_rate*N), replace=False)

                data_pose.extend(ds_data.read_coordinates(data_ids, field='pose'))
                data_shape.extend(ds_data.read_coordinates(data_ids, field='betas'))
                logger('randomly selected %d of %d data points in %s.'%(len(data_ids), len(ds_data), ds_name))
        else:
            logger('WARNING!!! HDF5 file not available %s'%ds_h5path)

    outdir = makepath(os.path.join(out_dir, split_name))

    outpath = os.path.join(outdir, 'data_pose.pt')
    torch.save(torch.tensor(np.asarray(data_pose, np.float32)), outpath)

    outpath = os.path.join(outdir, 'data_shape.pt')
    torch.save(torch.tensor(np.asarray(data_shape, np.float32)), outpath)

    logger('Len. split %s %d' %(split_name, len(data_pose)))
    logger('##############################################')


if __name__ == '__main__':
    # ['CMU', 'Transitions_mocap', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'MPI_mosh', 'MPI_HDM05', 'HumanEva', 'ACCAD', 'EKUT', 'SFU', 'KIT', 'H36M', 'TCD_handMocap', 'BioMotionLab_NTroje']

    dumpmode = 'pytorch'
    model_type = 'smpl'
    prior_type = 'VPoser'

    amass_dir = '/ps/project/amass/20190313/unified_results'
    out_dir = '/ps/project/smplbodyprior/BodyPrior/%s/data/20190313_amass_WO_CMU/%s/%s' % (prior_type, model_type, dumpmode)

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
    logger = log2file(os.path.join(out_dir, '%s.log' % (log_name)))
    global logger
    logger('Creating pytorch dataset at %s'%out_dir)

    vald_datasets = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
    test_datasets = ['Transitions_mocap', 'SSM_synced']
    train_datasets = ['MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'EKUT', 'KIT', 'TCD_handMocap', 'BioMotionLab_NTroje']
    # train_datasets = ['CMU', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'EKUT', 'KIT', 'TCD_handMocap', 'BioMotionLab_NTroje']
    # train_datasets = ['CMU', 'MPI_Limits', 'H3.6M']#cvpr19_initial
    train_datasets = list(set(train_datasets).difference(set(vald_datasets+test_datasets)))

    create_dataset_records(vald_datasets, amass_dir, out_dir, split_name='vald')
    create_dataset_records(test_datasets, amass_dir, out_dir, split_name='test')
    create_dataset_records(train_datasets, amass_dir, out_dir, split_name='train')

    script_name = os.path.basename(sys.argv[0])
    shutil.copy2(script_name, os.path.join(out_dir, script_name.replace('.py', '_%s.py' % log_name)))
