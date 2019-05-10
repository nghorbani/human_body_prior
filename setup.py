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
# 2019.05.10

try:
    from setuptools import setup, convert_path
except ImportError:
    from distutils.core import setup

setup(name='hposer',
      version='0.8.0',
      packages=['configer'],
      # package_data={'tests': ['./sample_settings.ini']},
      author='Nima Ghorbani',
      author_email='nima.gbani@gmail.com',
      maintainer='Nima Ghorbani',
      maintainer_email='nima.gbani@gmail.com',
      url='https://github.com/nghorbani/configer',
      description='Easy configuration of arguments in a python code!',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      license=u"GNU Affero General Public License, version 3",
      install_requires=['torch', 'tensorboardX', 'torchgeometry', 'opencv-python', 'scikit-image', 'configer', 'imageio', 'transforms3d'],
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