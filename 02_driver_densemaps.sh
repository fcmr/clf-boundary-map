projection_dir=$1
model_dir=$2
dataset_name=$3
grid_size=$4
num_per_cell=$5
binary=$6

test_id=$(date +"%Y%m%d_%H%M%S")
output_dir="densemaps_$test_id"

echo "Starting test id $test_id"

[ ! -d $output_dir ] && mkdir $output_dir

projections="$projection_dir/*.pkl"
models="$model_dir/*.pkl $model_dir/*.h5"

d=$dataset_name
g=$grid_size
n=$num_per_cell

for m in $models
do
	for p in $projections
	do
		echo "$d - $p (python run_densemaps.py -d $d -o $output_dir -p $p -m $m -g $g -n $n $binary)"
		python run_densemaps.py -d $d -o $output_dir -p $p -m $m -g $g -n $n $binary &

		while [ $(jobs | grep Running | wc -l ) -gt 7 ]
		do
			echo "--> $(jobs | grep Running | wc -l ) jobs running"
			sleep 10
		done		
	done
done

wait

echo "Start:  $test_id"
echo "Finish: $(date +"%Y%m%d_%H%M%S")"
