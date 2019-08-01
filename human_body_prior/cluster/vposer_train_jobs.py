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
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
#
# 2019.05.28

import os

def create_render_condor_subfile(sub_dir, log_dir, job_dir, jobId):
    job_path = os.path.join(job_dir, '%s.ini'%jobId)
    err_path = os.path.join(log_dir, '%s.err'%jobId)
    # out_path = os.path.join(log_dir, '%s.out'%jobId)
    # log_path = os.path.join(log_dir, '%s.log'%jobId)

    sub_contents = '''executable = /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/train_single.sh
arguments = %s
error = %s
request_cpus = 8
request_gpus = 2
request_memory = 64GB
requirements = TARGET.CUDAGlobalMemoryMb > 10240
queue'''%(job_path, err_path)
    sub_path = os.path.join(sub_dir, '%s.sub'%jobId)
    with open(sub_path, 'w') as f: f.writelines(sub_contents)
    return sub_path

def create_training_jobs(base_dir):
    '''pool size: -1 cluster, 1 local, >1 local pooling'''

    from configer import Configer
    from human_body_prior.tools.omni_tools import makepath

    job_dir = makepath('/is/cluster/nghorbani/train_vposer/jobs')
    sub_dir = makepath('/is/cluster/nghorbani/train_vposer/subs')
    cluster_logdir = makepath('/is/cluster/nghorbani/train_vposer/logs')

    args = {
        'model_type': 'smpl',
        'batch_size': 512,
        'latentD': 32,
        'num_neurons': 512,
        'n_workers': 10,
        'cuda_id':0,
        'base_lr': 5e-3,
        'test_only':False,

        'reg_coef': 1e-4,
        'kl_coef': 5e-3,
        'use_cont_repr': True,

        'ip_avoid': False,

        'adam_beta1': 0.9,
        'best_model_fname': None,
        'log_every_epoch': 2,
        'num_epochs': 100,
    }

    default_ps_fname = '/is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/train/vposer_smpl_defaults.ini'

    tasks = []
    job_paths = []
    for ds_name in ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'KIT','BML', 'EKUT']:
        expr_code = '004_00_WO_tcd_%s'%ds_name.lower()

        dataset_dir = '/ps/project/humanbodyprior/VPoser/data/004_00_WO_tcd_%s/smpl/pytorch/final_dsdir' % (ds_name.lower())

        if not os.path.exists(dataset_dir):
            print('dataset_dir does not exist: %s'%dataset_dir)
            continue

        work_dir = os.path.join(base_dir, expr_code)
        if os.path.exists(work_dir):
            print('work_dir already exists: %s'%work_dir)
            continue

        ps = Configer(default_ps_fname=default_ps_fname, work_dir=work_dir, dataset_dir=dataset_dir, expr_code=expr_code, **args)

        job_path = makepath(os.path.join(job_dir, '%s.ini'%expr_code), isfile=True)
        ps.dump_settings(fname=job_path)
        job_paths.append(job_path)

        create_render_condor_subfile(sub_dir, cluster_logdir, job_dir, expr_code)

        tasks.append(ps)
        # if len(tasks) > 0:break

    print('\n# of jobs', len(tasks))
    print('run on cluster: \nbash /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/vposer_train_condor.sh %s\n'%(sub_dir))

    # print('\nor run localy\nbash /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/vposer_train_local.sh %s %d\n'%(job_dir, 1))
    print('\nor run localy\n')
    for job in job_paths:
          print('bash /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/vposer_train_local.sh %s'%(job))

    return tasks

if __name__ == '__main__':
    base_dir = '/ps/project/humanbodyprior/VPoser/smpl/pytorch'
    tasks = create_training_jobs(base_dir)
