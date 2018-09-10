base_dir=$1

for d in lince03 lince09 lince13 lince21 lince25 lince27 lince29
do
    echo "Creating $base_dir/$d"
    [ ! -d $base_dir/$d ] && mkdir -p $base_dir/$d
done

i=0
j=0

declare -a servers=($(echo $base_dir/lince*))

ls ${base_dir}/*.pkl | awk -F/ '{print $2}' | awk -F_ '{print $2}' | sort -u | while read p
do
    echo "mv $base_dir/*_${p}*.pkl ${servers[$j]}"
    mv $base_dir/*_${p}*.pkl ${servers[$j]}
    i=$(expr $i + 1)

    if [ $i -gt 3 ]
    then
        i=0
        j=$(expr $j + 1)
    fi
done
