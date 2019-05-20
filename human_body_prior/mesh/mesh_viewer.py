import numpy as np
from human_body_prior.tools.omni_tools import colors
import trimesh
import pyrender
import sys
import cv2

__all__ = ['MeshViewer']

class MeshViewer(object):

    def __init__(self, width=1200, height=800, use_offscreen=True):
        super(MeshViewer, self).__init__()

        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors['white'], ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 2.5])
        self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(5.)
        else:
            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, viewport_size=self.figsize, cull_faces=False, run_in_thread=True)

        self.set_background_color(colors['white'])

    def set_background_color(self,color=colors['white']):
        self.scene.bg_color = color

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_meshes(self, meshes, group_name='static'):
        for node in self.scene.get_nodes():
            if node.name is not None and '%s-mesh'%group_name in node.name:
                self.scene.remove_node(node)

        for mid, mesh in enumerate(meshes):
            if isinstance(mesh, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(mesh)
            self.scene.add(mesh, '%s-mesh-%2d'%(group_name,mid))

    def set_static_meshes(self, meshes): self.set_meshes(meshes, 'static')
    def set_dynamic_meshes(self, meshes): self.set_meshes(meshes, 'dynamic')

    def _add_raymond_light(self):
        from pyrender.light import DirectionalLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(Node(
                light=DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes

    def use_raymond_lighting(self, intensity = 1.0):
        if not self.use_offscreen:
            sys.stderr.write('Interactive viewer already uses raymond lighting!\n')
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)#, parent_node=pc)

    def render(self):
        from pyrender.constants import RenderFlags

        flags = RenderFlags.SHADOWS_DIRECTIONAL #| RenderFlags.RGBA
        if self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth = self.viewer.render(self.scene, flags=flags)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        return color_img

    def save_snapshot(self, fname):
        if not self.use_offscreen:
            sys.stderr.write('Currently saving snapshots only works with offscreen renderer!\n')
            return
        color_img = self.render()
        cv2.imwrite(fname, color_img)

