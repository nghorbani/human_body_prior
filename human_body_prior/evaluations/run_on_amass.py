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
# 2019.05.28

import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath

VPOSER_DOWNLOAD_PAGE = 'https://smpl-x.is.tue.mpg.de/'


def evaluate_model(dataset_dir, vp_model, vp_ps, batch_size=5, save_upto_bnum=10, splitname='test'):
    from human_body_prior.data.dataloader import VPoserDS
    from human_body_prior.train.vposer_trainer import VPoserTrainer

    assert splitname in ['test', 'train', 'vald']
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds_name = dataset_dir.split('/')[-2]

    vp_model.eval()
    vp_model = vp_model.to(comp_device)

    with torch.no_grad():
        bm = BodyModel(vp_ps.body_model.bm_fname, batch_size=1, num_betas=16).to(comp_device)

    ds = VPoserDS(dataset_dir=os.path.join(dataset_dir, splitname))
    ds = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    outpath = os.path.join(vp_ps.logging.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(vp_ps.logging.best_model_fname).replace('.pt',''), '%s_samples'%splitname)
    print('dumping to %s'%outpath)

    for bId, dorig in enumerate(ds):
        dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

        imgpath = makepath(os.path.join(outpath, '%s-%03d.png' % (vp_ps.general.expr_id, bId)), isfile=True)
        VPoserTrainer.vis_results(dorig, bm, vp_model, imgpath, view_angles=[0, 180])#, view_angles = [0, 180, 90])

        if bId > save_upto_bnum:
            break


def evaluate_error(dataset_dir, vp_model, vp_ps, batch_size=512):
    from human_body_prior.data.dataloader import VPoserDS

    vp_model.eval()

    ds_name = dataset_dir.split('/')[-2]

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bm = BodyModel(vp_ps.body_model.bm_fname, batch_size=batch_size, num_betas=16).to(comp_device)
    vp_model = vp_model.to(comp_device)

    # from psbody.mesh import Mesh, MeshViewer
    # from human_body_prior.tools.omni_tools import colors
    # import time
    # mv = MeshViewer()

    final_errors = {}
    # for splitname in ['test']:
    for splitname in ['test', 'train', 'vald']:

        ds = VPoserDS(dataset_dir=os.path.join(dataset_dir, splitname))
        print('%s dataset size: %s'%(splitname,len(ds)))
        ds = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)#batchsize for bm is fixed so drop the last one

        loss_mean = []
        with torch.no_grad():
            for dorig in tqdm(ds):
                dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

                MESH_SCALER = 1000

                drec = vp_model(**dorig)
                for k in dorig: # whatever field is missing from drec copy from dorig
                    if k not in drec:
                        drec[k] = dorig[k]
                        if len(final_errors)==0 and len(loss_mean) == 0: 
                            print('Field %s is not predicted by the model and is copied from the original data.'%k)

                with torch.no_grad():
                    body_orig = bm(**dorig).v
                    body_rec = bm(**drec).v

                # body_orig_mesh = Mesh(c2c(body_orig[0]), c2c(bm.f), vc= colors['blue'])
                # body_rec_mesh = Mesh(c2c(body_rec[0]), c2c(bm.f), vc=colors['red'])
                # mv.set_dynamic_meshes([body_rec_mesh, body_orig_mesh])
                # time.sleep(0.2)

                # loss_mean.append(torch.mean(torch.sqrt(torch.pow((mesh_orig - mesh_rec)* MESH_SCALER, 2))))
                loss_mean.append(torch.mean(torch.abs(body_orig - body_rec)* MESH_SCALER))

        final_errors[splitname] = {'v2v_mae': float(c2c(torch.stack(loss_mean).mean()))}
        print(splitname, final_errors[splitname])

    outpath = makepath(os.path.join(vp_ps.logging.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(vp_ps.logging.best_model_fname).replace('.pt','.json')), isfile=True)
    with open(outpath, 'w') as f:
        json.dump(final_errors,f)

    return final_errors

def main(expr_dir=None, dataset_dir=None, batch_size=512):
    """Evaluate a trained VPoser on the splits of its dataset and return the v2v errors.

    :param expr_dir: directory of a trained VPoser (settings and snapshots), for example the
        VPoser download from https://smpl-x.is.tue.mpg.de/.
    :param dataset_dir: prepared VPoser dataset directory; defaults to the one recorded in the
        experiment settings.
    :raises ValueError: when ``expr_dir`` is not given.
    :raises FileNotFoundError: when ``expr_dir`` does not exist.
    """
    if expr_dir is None:
        raise ValueError('expr_dir is required: pass the directory of a trained VPoser, for example '
                         'the download from %s.' % VPOSER_DOWNLOAD_PAGE)
    if not os.path.isdir(expr_dir):
        raise FileNotFoundError('VPoser experiment directory not found at %s; download a trained VPoser '
                                'from %s.' % (expr_dir, VPOSER_DOWNLOAD_PAGE))

    from human_body_prior.models.vposer_model import VPoser
    from human_body_prior.tools.model_loader import load_model

    vp_model, vp_ps = load_model(expr_dir, model_code=VPoser, remove_words_in_model_weights='vp_model.',
                                 disable_grad=True)
    if dataset_dir is None:
        dataset_dir = vp_ps.logging.dataset_dir
    print('dataset_dir: %s' % dataset_dir)

    final_errors = evaluate_error(dataset_dir, vp_model, vp_ps, batch_size=batch_size)
    print('[%s] [DS: %s] -- %s' % (vp_ps.logging.best_model_fname, dataset_dir,
                                   ', '.join(['%s: %.2e' % (k, v['v2v_mae']) for k, v in final_errors.items()])))
    return final_errors


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a trained VPoser on its dataset splits.')
    parser.add_argument('--expr-dir', required=True,
                        help='directory of a trained VPoser (settings and snapshots), see %s' % VPOSER_DOWNLOAD_PAGE)
    parser.add_argument('--dataset-dir', default=None,
                        help='prepared VPoser dataset directory; default: the one in the experiment settings')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    main(expr_dir=args.expr_dir, dataset_dir=args.dataset_dir, batch_size=args.batch_size)
