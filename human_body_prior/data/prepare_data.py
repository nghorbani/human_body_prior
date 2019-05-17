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
from human_body_prior.train.train_vposer_smpl import VPoser
import shutil, sys
from torch.utils.data import Dataset

from datetime import datetime
import torch

def remove_Zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def dump_amass2pytroch(datasets, amass_dir, out_dir, split_name, logger = None, rnd_seed = 100):
    '''
    Select random number of frames from central 80 percent of each mocap sequence

    :param datasets:
    :param amass_dir:
    :param out_dir:
    :param split_name:
    :param logger
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

class VPoserDS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/
    The purpose of this interface is to make operations on an already obtained dataset [augmentation,...]"""

    def __init__(self, dataset_dir, dtype=torch.float32):

        self.ds = {'pose':torch.load(os.path.join(dataset_dir, 'data_pose.pt')),
                   'betas': torch.load(os.path.join(dataset_dir, 'data_betas.pt')),
                   'trans': torch.load(os.path.join(dataset_dir, 'data_trans.pt')),
                   'gender': torch.load(os.path.join(dataset_dir, 'data_gender.pt')),
                   }
        self.dtype = dtype


    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        pose_aa = self.ds['pose'][idx:idx+1].type(self.dtype)
        betas = self.ds['betas'][idx:idx+1].type(self.dtype)
        trans = self.ds['trans'][idx:idx+1].type(self.dtype)
        gender = self.ds['gender'][idx]

        pose_aa = pose_aa.view([1,1,-1,3])#[:,:,1:22] #removing root orient and fingers from body pose
        pose_matrot = VPoser.aa2matrot(pose_aa)

        sample = {'pose_aa': pose_aa[0].view(-1), #21*3
                  'pose_matrot': pose_matrot[0].view(-1), #21*9
                  'betas': betas[0],
                  'trans': trans[0],
                  'gender': gender,
                  }

        return sample

def prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=None):

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(vposer_datadir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % vposer_datadir)

    logger('Step 1: get SMPL parameters from AMASS by specific splits')
    vposer_interm_datadir = os.path.join(vposer_datadir, 'intermediate_files')

    for split_name, datasets in amass_splits.items():
        if os.path.exists(os.path.join(vposer_interm_datadir, split_name, 'data_pose.pt')): continue
        dump_amass2pytroch(datasets, amass_dir, vposer_interm_datadir, split_name=split_name, logger=logger)

    logger(
        'Step 2: augment data by adding different representation of poses, in parallel and save them into the final_data folder')
    from torch.utils.data import DataLoader

    final_dsdir = os.path.join(vposer_datadir, 'final_data')

    batch_size = 512
    max_num_epochs = 1  # how much augmentation we would get

    for split_name in amass_splits.keys():
        ds = VPoserDS(dataset_dir=os.path.join(vposer_interm_datadir, split_name))
        logger('%s has %d data points!' % (split_name, len(ds)))
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=False)

        data_all = {}

        for epoch_num in range(max_num_epochs):
            for bId, bData in enumerate(dataloader):
                for k, v in bData.items():
                    if k in data_all.keys():
                        data_all[k] = torch.cat([data_all[k], v], dim=0)
                    else:
                        data_all[k] = v

        for k, v in data_all.items():
            torch.save(v, makepath(os.path.join(vposer_datadir, 'final_data', split_name, '%s.pt' % k), isfile=True))

    logger('Dumped final pytorch dataset at %s' % final_dsdir)


if __name__ == '__main__':
    # ['CMU', 'Transitions_mocap', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'MPI_mosh', 'MPI_HDM05', 'HumanEva', 'ACCAD', 'EKUT', 'SFU', 'KIT', 'H36M', 'TCD_handMocap', 'BioMotionLab_NTroje']

    msg = ''' Using standard AMASS dataset preparation pipeline: 
    1) donwload all npz files. 
    2) convert npz files to pt ones so that parallel data augmentation is possible. 
    3) dump augmented results into final pt files that can be read in parallel for pytorch.'''

    dumpmode = 'pytorch'
    model_type = 'smpl'
    prior_type = 'VPoser'

    amass_dir = '/ps/project/amass/20190313/unified_results'
    vposer_datadir = makepath('/ps/project/smplbodyprior/BodyPrior/%s/data/0020_06_cmu_T3/%s/%s' % (prior_type, model_type, dumpmode))

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')

    shutil.copy2(sys.argv[0], os.path.join(vposer_datadir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % log_name)))

    logger = log2file(os.path.join(vposer_datadir, '%s.log' % (log_name)))
    logger(msg)

    amass_splits = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU'] #['EKUT', 'CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'KIT','BioMotionLab_NTroje']
    }
    amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

    prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=logger)