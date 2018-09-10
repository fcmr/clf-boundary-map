projection_dir=$1
model_dir=$2
dataset_name=$3
grid_size=$4
num_per_cell=$5
output_dir=$6
binary=$7

test_id=$(date +"%Y%m%d_%H%M%S")

echo "Starting test id $test_id"

[ ! -d $output_dir ] && mkdir $output_dir

projections=$(ls $projection_dir/${dataset_name}_*.pkl 2> /dev/null)
models=$(ls $model_dir/${dataset_name}_*.pkl $model_dir/${dataset_name}_*.h5 2> /dev/null)

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
