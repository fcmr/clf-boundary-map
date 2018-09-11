#!/bin/bash

test_id=$(date +"%Y%m%d_%H%M%S")
output_dir="densemaps_10class_$test_id"

[ ! -d $output_dir ] && mkdir $output_dir

declare -a servers=(lince03 lince09 lince13 lince21 lince25 lince27 lince29 lince32)

for s in ${servers[*]}
do
    nohup ssh $s "cd ~/src/clf-boundary-map && nohup ./11_run_jobs.sh $output_dir > $output_dir/${s}_nohup.log 2>&1 &" &
done

