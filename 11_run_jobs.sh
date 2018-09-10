module load Anaconda3-4.4.0
export CUDA_VISIBLE_DEVICES=

server=$(uname -n | awk -F. '{print $1}') 
projection_dir=projections_02class/$server
model_dir=classifiers_02class
dataset_name=fashionmnist
is_binary=-b

#for grid_size in 100 300 400
for grid_size in 10 20
do
    for num_per_cell in 1 2
    do
	output_dir="${server}_${dataset_name}_${model_dir}_${grid_size}_${num_per_cell}"
        echo "./02_driver_densemaps.sh $projection_dir $model_dir $dataset_name $grid_size $num_per_cell $output_dir $is_binary"
        ./02_driver_densemaps.sh $projection_dir $model_dir $dataset_name $grid_size $num_per_cell $output_dir $is_binary 
    done
done
