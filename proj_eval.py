#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from glob import glob

import joblib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import ParameterGrid

import mtsne
import projections
from metrics import *


def list_projections():
    return sorted(projections.all_projections.keys())


def list_datasets():
    return [os.path.basename(d) for d in sorted(glob('data/*'))]


def print_projections():
    print('Projections:' + ' '.join(list_projections()))


def print_datasets():
    print('Datasets:' + ' '.join(list_datasets()))


def load_dataset(dataset_name, is_binary=True):
    if is_binary:
        file_ext = '_sample_bin.npy'
    else:
        file_ext = '_sample.npy'

    data_dir = os.path.join('data', dataset_name)
    X = np.load(os.path.join(data_dir, 'X%s' % file_ext))
    y = np.load(os.path.join(data_dir, 'y%s' % file_ext))

    return X, y


def run_eval(dataset_name, projection_name, output_dir, is_binary):
    global DISTANCES

    X, y = load_dataset(dataset_name, is_binary)

    dc_results = dict()
    pq_results = dict()
    projected_data = dict()

#    dc_results['original'] = eval_dc_metrics(X=X, y=y, dataset_name=dataset_name, output_dir=output_dir)

    proj_tuple = projections.all_projections[projection_name]
    proj = proj_tuple[0]
    grid_params = proj_tuple[1]

    print(dataset_name, proj.__class__.__name__)
    grid = ParameterGrid(grid_params)

    for params in grid:
        id_run = proj.__class__.__name__ + '|' + str(params)
        proj.set_params(**params)

        print('-----------------------------------------------------------------------')
        print(dataset_name, id_run)

        try:
#            X_new, y_new, result = projections.run_projection(proj, X, y, id_run, dataset_name, output_dir)
            X_new, y_new = projections.run_projection(proj, X, y, id_run, dataset_name, output_dir)

#            pq_results[id_run] = result
            projected_data[id_run] = dict()
            projected_data[id_run]['X'] = X_new
            projected_data[id_run]['y'] = y_new
        except Exception as e:
            print('Error running %s: ' % id_run)
            print(str(e))
            print(sys.exc_info()[0])

    # results_to_dataframe(dc_results, dataset_name).to_csv(
    #     '%s/%s_%s_dc_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    # results_to_dataframe(pq_results, dataset_name).to_csv(
    #     '%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    joblib.dump(projected_data, '%s/%s_%s_%s_projected.pkl' % (output_dir, dataset_name, projection_name, str(is_binary)))
    # joblib.dump(DISTANCES, '%s/%s_%s_distance_files.pkl' % (output_dir, dataset_name, projection_name))
