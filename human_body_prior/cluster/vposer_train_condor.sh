#!/bin/bash
# given sub files wil try to run them on the cluster
#
sub_dir=$1
bid_amount=300

submitJob() {
  sub_file=$1
  echo $sub_file
  condor_submit_bid ${bid_amount} ${sub_file} -append 'requirements = CUDACapability>=7.0'
#   condor_submit ${sub_file} -append arguments\ =\ ${config_path}
#  echo condor_submit_bid ${bid_amount} ${sub_file} -append arguments\ =\ ${config_path}

}

echo ${sub_dir}
for entry in ${sub_dir}/004_00*.sub
do
  submitJob $entry
done


condor_q nghorbani