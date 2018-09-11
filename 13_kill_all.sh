#!/bin/bash

binary=$1

declare -a servers=(lince03 lince09 lince13 lince21 lince25 lince27 lince29 lince32)

ps -ef | grep 11_run | grep -v grep | awk '{print $2}' | xargs kill

for s in ${servers[*]}
do
    echo $s
    ssh $s "ps -ef | grep '02_driver_densemaps' | grep -v grep | awk '{print \$2}' | xargs kill" 
    ssh $s "ps -ef | grep 'python run_densemaps' | grep -v grep | awk '{print \$2}' | xargs kill" 
done

