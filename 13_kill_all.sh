#!/bin/bash

binary=$1

declare -a servers=(lince03 lince09 lince13 lince21 lince25 lince27 lince29 lince32)

for s in ${servers[*]}
do
    echo $s
    ssh $s "ps -ef | grep 'python run_' | grep -v grep | awk '{print \$2}' | xargs kill" 
done

