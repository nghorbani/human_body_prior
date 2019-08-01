#!/bin/bash
deactivate
conda activate py37cluster

#job_dir=$1
#n_jobs=$2
job_path=$1
doJob() {
  config_path=$1
  bash /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/train_single.sh ${config_path}
#  python3 /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/train/vposer_smpl.py --config_path ${config_path}

}


doJob $job_path

#for entry in ${job_dir}/*.ini
#do
#echo $entry
#  doJob $entry
#done

#find ${job_dir} -name *.ini | parallel -I% --max-args 1 bash /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/train_single.sh %;
#find ${job_dir} -name *.ini | parallel -j ${n_jobs} -I% --max-args 1 bash /is/ps2/nghorbani/code-repos/human_body_prior/human_body_prior/cluster/train_single.sh %;
