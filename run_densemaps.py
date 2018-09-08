import argparse
import os
import numpy as np

import densemap
import proj_eval

if __name__ == '__main__':
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Dense Map Runner')

    parser.add_argument('-d', type=str, help='dataset name')
    parser.add_argument('-o', type=str, help='output directory (must exist)')
    parser.add_argument('-p', type=str, help='projection name')
    parser.add_argument('-m', type=str, help='model name')
    parser.add_argument('-b', action='store_true', help='is binary problem?')
    parser.add_argument('-g', type=int, default=100, help='grid size')
    parser.add_argument('-n', type=int, default=2, help='samples per cell')
    args, unknown = parser.parse_known_args()

    if args.d is None:
        proj_eval.print_datasets()
        exit()

    dataset_name = args.d
    output_dir = args.o
    projection_name = args.p
    model_name = args.m
    is_binary = args.b

    grid_size = int(args.g)
    num_per_cell = int(args.n)

    if not os.path.exists(output_dir):
        print('Directory %s not found' % output_dir)
        exit(1)

    densemap.create_densemap(dataset_name, output_dir, model_name, projection_name, is_binary, grid_size, num_per_cell)

    # models = glob(os.path.join(classifiers_dir, '%s_model_*_%s.*' % (dataset_name, str(is_binary))))
    # projections = glob(os.path.join(projections_dir, '%s_*_%s_projected.pkl' % (dataset_name, str(is_binary))))
