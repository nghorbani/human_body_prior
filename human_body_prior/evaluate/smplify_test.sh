#!/usr/bin/env bash

conda activate py37cluster
#FOCAL_LENGTH=1500
vposer_workdir="/ps/project/common/vposer/smpl/0020_0700R10_T2"
#vposer_workdir="/ps/project/humanbodyprior/VPoser/smpl/pytorch/004_00_WO_tcd_handmocap"


python3 /ps/project/common/smplifypp/public/smplifyx/main.py --config /ps/project/common/smplifypp/public/cfg_files/smplx_debug.yaml \
--data_folder /ps/project/common/smplifypp/EHT_data \
--output_folder $vposer_workdir'/evaluations/smplifypp' \
--interpenetration=True \
--visualize=True \
--gender="male" \
--model_folder /ps/project/common/smplifypp/models \
--vposer_ckpt $vposer_workdir