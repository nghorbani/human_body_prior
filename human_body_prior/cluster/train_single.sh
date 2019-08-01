#!/bin/bash

#deactivate
#module load cuda/9.2
#conda activate py37cluster

config_path=$1

#cd /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/train
#python3 -m vposer_smpl


/is/ps2/nghorbani/opt/anaconda3/envs/py37cluster/bin/python /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/train/vposer_smpl.py --config_path ${config_path}
#python3 /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/train/vposer_smpl.py --config_path ${config_path}
