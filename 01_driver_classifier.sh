#!/bin/bash

binary=$1

test_id=$(date +"%Y%m%d_%H%M%S")
output_dir="classifiers_$test_id"

datasets=$(python run_projections.py | head -1 | awk -F: '{print $2}')

echo "Starting test id $test_id"

[ ! -d $output_dir ] && mkdir $output_dir

for d in $datasets
do
	echo "$d (python run_classifiers.py -d $d -o $output_dir $binary)"
	python run_classifiers.py -d $d -o $output_dir $binary &
done

wait

echo "Start:  $test_id"
echo "Finish: $(date +"%Y%m%d_%H%M%S")"
