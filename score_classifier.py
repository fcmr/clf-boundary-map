import os
import argparse

import classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification Scorer')

    parser.add_argument('-d', type=str, help='dataset name')
    parser.add_argument('-b', action='store_true', help='is binary problem?')
    parser.add_argument('-c', type=str, help='classifier directory (must exist)')
    args, unknown = parser.parse_known_args()

    dataset_name = args.d
    classifier_dir = args.c
    is_binary = args.b

    if args.d is None:
        print('Error: must specify dataset')
        exit(1)

    if not os.path.exists(classifier_dir):
        print('Directory %s not found' % classifier_dir)
        exit(1)

    classifier.score_classifiers(dataset_name, classifier_dir, is_binary)
