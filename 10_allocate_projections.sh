#!/bin/bash

base_dir=$1

if [ ! -d "$base_dir" ]
then
    echo "Usage: 10_allocate_projections.sh <projections_dir>"
    exit 1
fi

declare -a servers=(lince03 lince09 lince13 lince21 lince25 lince27 lince29 lince32)

for d in ${servers[*]}
do
    echo "Creating $base_dir/$d"
    [ ! -d $base_dir/$d ] && mkdir -p $base_dir/$d
done

i=0

ls ${base_dir}/*.pkl | awk -F/ '{print $2}' | awk -F_ '{print $2}' | sort -u | while read p
do
    echo "mv $base_dir/*_${p}*.pkl $base_dir/${servers[$i]}"
    mv $base_dir/*_${p}*.pkl $base_dir/${servers[$i]}

    i=$(expr $i + 1)

    if [ $i -eq ${#servers[@]} ]
    then
        i=0
    fi
done

