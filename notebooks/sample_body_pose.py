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
# 2019.05.10

__all__ = ['sample_vposer']
import os
import numpy as np

import trimesh

from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors, makepath
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.visualization_tools import imagearray2file, smpl_params2ply

def dump_vposer_samples(bm, pose_body, out_imgpath=False, save_ply=False):
    '''
    
    :param bm: the BodyModel instance
    :param pose_body: Nx63 will pose the body
    :param out_imgpath: the final png path
    :param save_ply: if True will dump as ply files
    :return: 
    '''

    view_angles = [0, 90, -90]
    imw, imh = 400,400
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    images = np.zeros([len(view_angles), len(pose_body),1, imw, imh, 3])
    for cId in range(0, len(pose_body)):

        bm.pose_body.data[:] = bm.pose_body.new(pose_body[cId].reshape(-1))

        body_mesh = trimesh.Trimesh(vertices=c2c(bm().v[0]), faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))

        for rId, angle in enumerate(view_angles):
            apply_mesh_tranfsormations_([body_mesh],
                                        trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
            mv.set_meshes([body_mesh], group_name='static')
            images[rId, cId,0] = mv.render()
            apply_mesh_tranfsormations_([body_mesh],
                                        trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

    if out_imgpath:
        imagearray2file(images, out_imgpath)

        np.savez(out_imgpath.replace('.png', '.npz'), pose=pose_body)

        if save_ply:
            im_id = os.path.basename(out_imgpath).split('.')[0]
            out_dir = makepath(os.path.join(os.path.dirname(out_imgpath), '%s_ply'%im_id))
            smpl_params2ply(bm, out_dir=out_dir, pose_body=pose_body)

        print('Saved image: %s' % out_imgpath)

    return images

def sample_vposer(expr_dir, bm, num_samples=5, vp_model='snapshot'):
    from human_body_prior.tools.omni_tools import id_generator, makepath
    from human_body_prior.tools.model_loader import load_vposer
    from human_body_prior.tools.omni_tools import copy2cpu

    vposer_pt, ps = load_vposer(expr_dir, vp_model=vp_model)

    sampled_pose_body = copy2cpu(vposer_pt.sample_poses(num_poses=num_samples))

    out_dir = makepath(os.path.join(ps.work_dir, 'evaluations', 'pose_generation'))
    out_imgpath = os.path.join(out_dir, '%s.png' % id_generator(6))

    dump_vposer_samples(bm, sampled_pose_body, out_imgpath)
    print('Dumped samples at %s'%out_dir)
    return sampled_pose_body