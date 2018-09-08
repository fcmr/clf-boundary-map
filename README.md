# clf-boundary-map

- Setup:

```
conda install umap-learn keras wget numpy pandas scikit-learn
conda install -c conda-forge shogun

git clone https://github.com/DimitryUlyanov/Multicore-TSNE.git
cd Multicore-TSNE
pip install .
```

- Getting the datasets (will create a folder named "data")

```
python get_data.py
```

- List available datasets
```
python run_projections.py | head -1 | awk -F: '{print $2}'
```

- List available projections
```
python run_projections.py | tail -1 | awk -F: '{print $2}'
```

- Running a single projection
```
python run_projections.py -d <dataset_name> -p <projection_name> -o <output_dir (must exist)> [-b (to indicate that it's a binary problem)]
```

- Running all classifiers for a single dataset
```
python run_classifiers.py -d <dataset_name> -o <output_dir (must exist)> [-b (to indicate that it's a binary problem)]
```

- Running a single densemap
```
python run_densemaps.py -d <dataset_name> -p <projection_dir (location of pkl files)> -m <model_dir (location of pkl and h5 files) -g <grid_size> -n <num_per_cell> -o <output_dir (must exist)> [-b (to indicate that it's a binary problem)]
```

- Running all projections
```
./00_driver_projections.sh [-b]
```

- Running all classifiers for all datasets
```
./01_driver_classifiers.sh [-b]
```

- Running all densemaps for a single dataset
```
./02_driver_densemaps.sh <projection_dir> <model_dir> <dataset_name> <grid_size> <num_per_cell> [-b]
```

- Add/change projections: see projections.py
