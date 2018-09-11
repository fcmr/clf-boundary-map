#!/bin/bash

declare -a servers=(lince03 lince09 lince13 lince21 lince25 lince27 lince29 lince32)

for s in ${servers[*]}
do
    echo "$s"
    nohup ssh $s 'cd ~/src/clf-boundary-map && nohup ./11_run_jobs.sh > $(uname -n).log 2>&1 &' &
done


