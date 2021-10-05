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


from setuptools import setup, find_packages
from glob import glob
setup(name='human_body_prior',
      version='2.2.2.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      data_files=[('human_body_prior/support_data', glob('support_data/*.*'))],


      author='Nima Ghorbani',
      author_email='nghorbani@tue.mpg.de',
      maintainer='Nima Ghorbani',
      maintainer_email='nghorbani@tue.mpg.de',
      url='https://github.com/nghorbani/human_body_prior',
      description='Variational human pose prior for human pose synthesis and estimation.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=[],
      dependency_links=[],
      classifiers=[
          "Intended Audience :: Research",
          "Natural Language :: English",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: BSD",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7", ],
      )
