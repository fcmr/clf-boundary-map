#!/bin/bash

module load Anaconda3-4.4.0
export CUDA_VISIBLE_DEVICES=

server=$(uname -n | awk -F. '{print $1}') 
projection_dir=projections_10class/$server
model_dir=classifiers_10class
output_dir=$1

datasets=fashionmnist

test_id=$(date +"%Y%m%d_%H%M%S")

for dataset_name in $datasets
do
    #for grid_size in 100 300 400
    for grid_size in 400
    do
        for num_per_cell in 5
        do
            ./02_driver_densemaps.sh $projection_dir $model_dir $dataset_name $grid_size $num_per_cell $output_dir >> ${output_dir}/${server}.log 2>&1
        done
    done
done

echo "Start:  $test_id"
echo "Finish: $(date +"%Y%m%d_%H%M%S")"

