import argparse
import os

import proj_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Projection Survey Runner')

    parser.add_argument('-d', type=str, help='dataset name')
    parser.add_argument('-b', action='store_true', help='is binary problem?')
    parser.add_argument('-p', type=str, help='projection name')
    parser.add_argument('-o', type=str, help='output directory (must exist)')
    args, unknown = parser.parse_known_args()

    if args.d is None:
        proj_eval.print_datasets()
        proj_eval.print_projections()
        exit()

    dataset_name = args.d
    projection_name = args.p
    output_dir = args.o
    is_binary = args.b

    if not os.path.exists(output_dir):
        print('Directory %s not found' % output_dir)
        exit(1)

    proj_eval.run_eval(dataset_name, projection_name, output_dir, is_binary)
