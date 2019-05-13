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
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.omni_tools import euler2em, em2euler
import shutil, sys

import tables as pytables
from datetime import datetime
import torch

def remove_Zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def create_dataset_records_V1(datasets, amass_dir, out_dir, split_name, rnd_seed = 100):
    '''
    Select random number of frames from all poses within a dataset
    one caveat is that more standing poses will be selected
    :param datasets:
    :param amass_dir:
    :param out_dir:
    :param split_name:
    :param rnd_seed:
    :return:
    '''

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

def create_dataset_records_V2(datasets, amass_dir, out_dir, split_name, logger = None, rnd_seed = 100):
    '''
    Select random number of frames from central 80 percent of each mocap sequence
    This is to remedy the issue in V1 and should be tested.

    :param datasets:
    :param amass_dir:
    :param out_dir:
    :param split_name:
    :param rnd_seed:
    :return:
    '''
    import glob
    from tqdm import tqdm

    assert split_name in ['train', 'vald', 'test']
    np.random.seed(rnd_seed)

    makepath(out_dir, isfile=False)

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(out_dir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % out_dir)

    keep_rate = 0.3 # 30 percent

    data_pose = []
    data_betas = []
    data_gender = []
    data_trans = []

    for ds_name in datasets:
        npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*.npz'))
        logger('randomly selecting data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            cdata = np.load(npz_fname)
            N = len(cdata['poses'])

            cdata_ids = np.random.choice(list(range(int(0.1*N), int(0.9*N),1)), int(keep_rate*0.8*N), replace=False)
            if len(cdata_ids)<1: continue

            data_pose.extend(cdata['poses'][cdata_ids].astype(np.float32))
            data_trans.extend(cdata['trans'][cdata_ids].astype(np.float32))
            data_betas.extend(np.repeat(cdata['betas'][np.newaxis].astype(np.float32), repeats=len(cdata_ids), axis=0))
            data_gender.extend([{'male':-1, 'neutral':0, 'female':1}[str(cdata['gender'].astype(np.str))] for _ in cdata_ids])

    outdir = makepath(os.path.join(out_dir, split_name))

    assert len(data_pose) != 0

    outpath = os.path.join(outdir, 'data_pose.pt')
    torch.save(torch.tensor(np.asarray(data_pose, np.float32)), outpath)

    outpath = os.path.join(outdir, 'data_betas.pt')
    torch.save(torch.tensor(np.asarray(data_betas, np.float32)), outpath)

    outpath = os.path.join(outdir, 'data_trans.pt')
    torch.save(torch.tensor(np.asarray(data_trans, np.float32)), outpath)

    outpath = os.path.join(outdir, 'data_gender.pt')
    torch.save(torch.tensor(np.asarray(data_gender, np.int32)), outpath)

    logger('Len. split %s %d' %(split_name, len(data_pose)))

if __name__ == '__main__':
    # ['CMU', 'Transitions_mocap', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'MPI_mosh', 'MPI_HDM05', 'HumanEva', 'ACCAD', 'EKUT', 'SFU', 'KIT', 'H36M', 'TCD_handMocap', 'BioMotionLab_NTroje']

    msg = '''Trying dataset preparation funtion V2. The new sampling is exxptected to include less number of standing poses of the subject. 
Before random samples from all the dataset were being picked but now random samples from each mocap sequence"s central 80 percent is picked'''

    dumpmode = 'pytorch'
    model_type = 'smpl'
    prior_type = 'VPoser'

    amass_dir = '/ps/project/amass/20190313/unified_results'
    out_dir = '/ps/project/smplbodyprior/BodyPrior/%s/data/20190313_cmu_T2/%s/%s' % (prior_type, model_type, dumpmode)

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')

    logger = log2file(os.path.join(out_dir, '%s.log' % (log_name)))
    logger('Creating pytorch dataset at %s'%out_dir)
    logger(msg)

    vald_datasets = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
    test_datasets = ['Transitions_mocap', 'SSM_synced']
    # train_datasets = ['MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'EKUT', 'KIT', 'TCD_handMocap', 'BioMotionLab_NTroje']
    train_datasets = ['CMU']
    # train_datasets = ['CMU', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'EKUT', 'KIT', 'TCD_handMocap', 'BioMotionLab_NTroje']
    # train_datasets = ['CMU', 'MPI_Limits', 'H3.6M']#cvpr19_initial
    train_datasets = list(set(train_datasets).difference(set(vald_datasets+test_datasets)))

    create_dataset_records_V2(vald_datasets, amass_dir, out_dir, split_name='vald', logger=logger)
    create_dataset_records_V2(test_datasets, amass_dir, out_dir, split_name='test', logger=logger)
    create_dataset_records_V2(train_datasets, amass_dir, out_dir, split_name='train', logger=logger)

    script_name = os.path.basename(sys.argv[0])
    shutil.copy2(script_name, os.path.join(out_dir, script_name.replace('.py', '_%s.py' % log_name)))
