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
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <http://arxiv.org/abs/1904.05866>

#
#
# Code Developed by:
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
#
# 2019.05.10

from setuptools import setup, find_packages

setup(name='human_body_prior',
      version='0.8.1',
      packages = find_packages(),
      # packages=['human_body_prior', 'human_body_prior/data', 'human_body_prior/tutorials', 'human_body_prior/models', 'human_body_prior/tools'],
      # package_data={'tests': ['./samples']},
      author='Nima Ghorbani',
      author_email='nima.gbani@gmail.com',
      maintainer='Nima Ghorbani',
      maintainer_email='nima.gbani@gmail.com',
      url='https://github.com/nghorbani/HumanBodyPrior',
      description='Variational human pose prior for human pose synthesis and estimation.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      license=u"GNU Affero General Public License, version 3",
      install_requires=['torch>=1.1.0', 'tensorboardX>=1.6', 'torchgeometry>=0.1.2', 'opencv-python>=4.1.0.25',
                        'scikit-image>=0.15.0', 'configer>=1.2.3', 'imageio>=2.5.0', 'transforms3d>=0.3.1', 'trimesh'],
      classifiers=[
          "Intended Audience :: Developers",
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          "Natural Language :: English",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: BSD",
          "Operating System :: POSIX :: Linux",
          "Operating System :: Microsoft :: Windows",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",],
      )