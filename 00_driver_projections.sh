binary=$1

test_id=$(date +"%Y%m%d_%H%M%S")
output_dir="projections_$test_id"

datasets=$(python run_projections.py | head -1 | awk -F: '{print $2}')
projections=$(python run_projections.py | tail -1 | awk -F: '{print $2}')

echo "Starting test id $test_id"

[ ! -d $output_dir ] && mkdir $output_dir

# for d in $datasets
# do
# 	echo "$d - PCA"
# 	python run_projections.py -d $d -p PCA -o $output_dir $binary &

# 	while [ $(jobs | grep Running | wc -l ) -gt 3 ]
# 	do
# 		echo "--> $(jobs | grep Running | wc -l ) jobs running"
# 		sleep 10
# 	done	
# done

# wait

for d in $datasets
do
	for p in $projections
	do
		# if [ "$p" = "PCA" ]
		# then
		# 	continue
		# fi

		echo "$d - $p (python run_projections.py -d $d -p $p -o $output_dir $binary)"
		python run_projections.py -d $d -p $p -o $output_dir $binary &

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
