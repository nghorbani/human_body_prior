"""Map SMPL-X and SMPL-H body parts to vertex ids from the models' blend weights.

The model files are licence-gated downloads: SMPL-X from https://smpl-x.is.tue.mpg.de/ and
SMPL-H from https://mano.is.tue.mpg.de/. Pass their paths explicitly; there are no defaults.
Run as a script for the command line interface, for example::

    python -m human_body_prior.tools.bodypart2vertexid smplh --bm-fname model.npz --out part2vids.npz
"""
import argparse
import os

import numpy as np

SMPLX_DOWNLOAD_PAGE = 'https://smpl-x.is.tue.mpg.de/'
SMPLH_DOWNLOAD_PAGE = 'https://mano.is.tue.mpg.de/'


def require_file(path, what, download_page):
    """Return ``path`` when it names an existing file, otherwise raise a descriptive error.

    :raises ValueError: when ``path`` is None.
    :raises FileNotFoundError: when ``path`` does not exist.
    """
    if path is None:
        raise ValueError(f'{what} is required: pass the path of the file downloaded from {download_page}.')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{what} not found at {path}; download it from {download_page}.')
    return path


def find_handVertexIDs(blend_weights, all_partIds, interested_partIds):

    segm = np.argmax(blend_weights, axis=1)               # n_vertex
    num2part = {v: k for k, v in all_partIds.items()}    # n_joints
    print(num2part)
    vert2part = [num2part[i] for i in segm]                 # 6890

    PartLABELs = [num2part[ii] for ii in interested_partIds]
    VertexIDs = [ii for ii in range(len(vert2part)) if vert2part[ii] in PartLABELs]

    # test
    allOK = all(vert2part[VertexID] in PartLABELs for VertexID in VertexIDs)
    print('allOK =', allOK)
    print('isSorted =', all(VertexIDs[i] <= VertexIDs[i+1] for i in range(len(VertexIDs)-1)))

    return VertexIDs


def _part_ids_from_joints(part_joints):
    all_partids = {}
    for bk, jids in part_joints.items():
        for jid in jids:
            all_partids['%s_%02d' % (bk, jid)] = jid
            print('%s_%02d' % (bk, jid))
    return all_partids


def _show_parts(body_v, part2vids, part_color_names, snapshot_fname=None):
    """Colour the parts in a psbody MeshViewer; needs the psbody extra and body_visualizer."""
    from body_visualizer.tools.vis_tools import colors
    from psbody.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewer

    mv = MeshViewer(keepalive=True)
    meshes = [Mesh(v=body_v[part2vids[partname]], f=[], vc=colors[color_name])
              for partname, color_name in part_color_names.items()]
    mv.set_static_meshes(meshes)
    if snapshot_fname is not None:
        mv.save_snapshot(snapshot_fname)


def smplx_part_ids(bm_fname=None, joints_fname=None, out_fname=None, show=False):
    """Compute the SMPL-X part to vertex-id mapping.

    :param bm_fname: SMPL-X model file (model.npz) from https://smpl-x.is.tue.mpg.de/.
    :param joints_fname: npz file with a ``joints`` array that poses the model, for example the
        downsampled SMPL-X model of the same site.
    :param out_fname: where to save the mapping as npz; None keeps it in memory only.
    :param show: open a psbody MeshViewer with the coloured parts.
    :return: dict part name -> sorted vertex ids, plus ``'all'``.
    """
    require_file(bm_fname, 'SMPL-X body model file', SMPLX_DOWNLOAD_PAGE)
    require_file(joints_fname, 'SMPL-X joints file', SMPLX_DOWNLOAD_PAGE)

    import torch

    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    bm = BodyModel(bm_fname=bm_fname)
    joints = torch.from_numpy(np.load(joints_fname)['joints'])

    smplx_partids = {
                    'body': [0,1,2,3,4,5,6,9,13,14,16,17,18,19],
                    'face': [12, 15, 22],
                    'eyeball': [23, 24],
                     'leg': [4, 5, 7, 8, 10, 11],
                     'arm': [18, 19, 20, 21],
                    'handl': [20] + list(range(25, 40)),
                    'handr': [21] + list(range(40, 55)),
                    'footl': [7,10],
                    'footr': [8,11],
                     'ftip': [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]

                     }

    all_partids = _part_ids_from_joints(smplx_partids)

    body_part_vc = {'body': 'yellow', 'arm': 'orange', 'face': 'green', 'leg': 'green',
                    'ftip': 'white',
                    'footl': 'blue', 'footr': 'blue',
                    'handl': 'red', 'handr': 'orange',
                    }

    part2vids = {}
    for partname, partids in smplx_partids.items():
        vertex_ids = find_handVertexIDs(c2c(bm.weights), all_partids, partids)
        vertex_ids = np.array(sorted(vertex_ids))
        part2vids[partname] = vertex_ids

    body_v = c2c(bm(joints=joints).v[0])

    part2vids['all'] = np.arange(0, body_v.shape[0])

    if out_fname is not None:
        np.savez(out_fname, **part2vids)
    if show:
        _show_parts(body_v, part2vids, body_part_vc)
    return part2vids


def smplh_part_ids(bm_fname=None, out_fname=None, snapshot_fname=None, show=False):
    """Compute the SMPL-H part to vertex-id mapping.

    :param bm_fname: SMPL-H model file (model.npz) from https://mano.is.tue.mpg.de/.
    :param out_fname: where to save the mapping as npz; None keeps it in memory only.
    :param snapshot_fname: where to save a rendering of the coloured parts; implies ``show``.
    :param show: open a psbody MeshViewer with the coloured parts.
    :return: dict part name -> sorted vertex ids, plus ``'all'``.
    """
    require_file(bm_fname, 'SMPL-H body model file', SMPLH_DOWNLOAD_PAGE)

    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    bm = BodyModel(bm_fname=bm_fname)

    smplx_partids = {'body': [0, 1, 2, 3, 4, 5, 6, 9, 13, 14, 16, 17, 18, 19,22,23,24,],
                     'face': [15, 12],
                     'handl': [20] + list(range(22, 37)),
                     'handr': [21] + list(range(37, 52)),
                     'footl': [7, 10],
                     'footr': [8, 11],
                     }
    all_partids = _part_ids_from_joints(smplx_partids)

    body_part_vc = {k: v for k, v in {'body': 'yellow',
                                      'face': 'green',
                                      'footl': 'brown', 'footr': 'blue',
                                      'handl': 'pink', 'handr': 'orange',
                                      }.items() if k in smplx_partids}
    part2vids = {}
    for partname, partids in smplx_partids.items():
        vertex_ids = find_handVertexIDs(c2c(bm.weights), all_partids, partids)
        vertex_ids = np.array(sorted(vertex_ids))
        part2vids[partname] = vertex_ids

    body_v = c2c(bm().v[0])

    part2vids['all'] = np.arange(0, body_v.shape[0])
    if out_fname is not None:
        np.savez(out_fname, **part2vids)
    if show or snapshot_fname is not None:
        _show_parts(body_v, part2vids, body_part_vc, snapshot_fname=snapshot_fname)
    return part2vids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='model', required=True)
    px = sub.add_parser('smplx', help='SMPL-X part to vertex ids')
    px.add_argument('--bm-fname', required=True, help='SMPL-X model.npz (%s)' % SMPLX_DOWNLOAD_PAGE)
    px.add_argument('--joints-fname', required=True, help='npz with a joints array that poses the model')
    px.add_argument('--out', default=None, help='output npz; omit to only print')
    px.add_argument('--show', action='store_true', help='open a psbody MeshViewer')
    ph = sub.add_parser('smplh', help='SMPL-H part to vertex ids')
    ph.add_argument('--bm-fname', required=True, help='SMPL-H model.npz (%s)' % SMPLH_DOWNLOAD_PAGE)
    ph.add_argument('--out', default=None, help='output npz; omit to only print')
    ph.add_argument('--snapshot', default=None, help='save a rendering of the parts to this file')
    ph.add_argument('--show', action='store_true', help='open a psbody MeshViewer')
    args = parser.parse_args(argv)
    if args.model == 'smplx':
        smplx_part_ids(bm_fname=args.bm_fname, joints_fname=args.joints_fname, out_fname=args.out, show=args.show)
    else:
        smplh_part_ids(bm_fname=args.bm_fname, out_fname=args.out, snapshot_fname=args.snapshot, show=args.show)


if __name__ == '__main__':
    main()
