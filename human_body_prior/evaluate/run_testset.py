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
# Vassilis Choutas <https://ps.is.tuebingen.mpg.de/employees/vchoutas> for ContinousRotReprDecoder
#
# 2019.05.28

import os

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.tools.omni_tools import makepath, id_generator
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.train.vposer_smpl import VPoserTrainer
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import DataLoader
import torch

def save_testset_samples(dataset_dir, vposer_model, ps, batch_size=5, save_upto_bnum=10):
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vposer_model.eval()
    vposer_model = vposer_model.to(comp_device)


    bm_path = '/ps/project/common/moshpp/smplh/locked_head/neutral/model.npz'
    bm = BodyModel(bm_path, 'smplh', batch_size=batch_size, use_posedirs=True).to(comp_device)

    ds_test = VPoserDS(dataset_dir=os.path.join(dataset_dir, 'test'))
    ds_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=False)

    test_savepath = os.path.join(ps.work_dir, 'evaluations', os.path.basename(ps.best_model_fname).replace('.pt',''))

    for bId, dorig in enumerate(ds_test):
        dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

        imgpath = makepath(os.path.join(test_savepath, '%s.png' % (id_generator(5))), isfile=True)
        VPoserTrainer.vis_results(dorig, bm=bm, vposer_model=vposer_model, imgpath=imgpath)
        if bId> save_upto_bnum: break


def evaluate_test_error(dataset_dir, vposer_model, ps, batch_size=512):
    vposer_model.eval()

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bm_path = '/ps/project/common/moshpp/smplh/locked_head/neutral/model.npz'
    bm = BodyModel(bm_path, 'smplh', batch_size=batch_size, use_posedirs=True).to(comp_device)

    vposer_model = vposer_model.to(comp_device)

    ds_test = VPoserDS(dataset_dir=os.path.join(dataset_dir, 'test'))
    ds_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=True)

    loss = []
    with torch.no_grad():
        for dorig in ds_test:
            dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

            MESH_SCALER = 1000

            drec = vposer_model(dorig['pose'], input_type='aa', output_type='aa')

            mesh_orig = bm(pose_body=dorig['pose'].view(ps.batch_size, -1)).v * MESH_SCALER
            mesh_rec = bm(pose_body=drec['pose'].view(ps.batch_size, -1)).v * MESH_SCALER
            loss.append(torch.mean(torch.abs(mesh_orig - mesh_rec)))

    v2v_mae = c2c(torch.stack(loss).mean())
    return v2v_mae

if __name__ == '__main__':
    dataset_dir= '/ps/project/humanbodyprior/VPoser/data/004_00_amass/smpl/pytorch/final_dsdir'
    expr_basedir = '/ps/project/humanbodyprior/VPoser/smpl/pytorch'

    for ds_name in ['amass', 'CMU','EKUT', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'KIT','BML','TCD_handMocap']:
        for prex in ['', '_WO']:
            expr_code = '004_00' + prex + '_%s'%ds_name.lower()
            expr_dir = os.path.join(expr_basedir, expr_code)
            if not os.path.exists(expr_dir):
                print('%s does not exist'%expr_dir)
                continue

            vposer_model, vposer_ps = load_vposer(expr_dir, vp_model='snapshot')

            save_testset_samples(dataset_dir, vposer_model, vposer_ps, batch_size=5, save_upto_bnum=10)
            v2v_mae = evaluate_test_error(dataset_dir, vposer_model, vposer_ps, batch_size=512)
            print('[%s] v2v_mae = %.2e' % (vposer_ps.best_model_fname, v2v_mae))
