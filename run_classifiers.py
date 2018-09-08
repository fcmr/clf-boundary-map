import os
import argparse

import classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification Runner')

    parser.add_argument('-d', type=str, help='dataset name')
    parser.add_argument('-b', action='store_true', help='is binary problem?')
    parser.add_argument('-o', type=str, help='output directory (must exist)')
    args, unknown = parser.parse_known_args()

    dataset_name = args.d
    output_dir = args.o
    is_binary = args.b

    if args.d is None:
        print('Error: must specify dataset')
        exit(1)

    if not os.path.exists(output_dir):
        print('Directory %s not found' % output_dir)
        exit(1)

    classifier.run_classifiers(dataset_name, output_dir, is_binary)
