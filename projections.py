#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import umap
import numpy as np
from sklearn import (decomposition, discriminant_analysis, manifold,
                     random_projection)
from time import perf_counter
import mtsne
import sgtk
import vp
import metrics
import joblib

all_projections = dict()
projections_to_save = ['PCA', 'KernelPCASigmoid', 'KernelPCALinear', 'KernelPCAPoly', 'KernelPCARbf', 'UMAP']

def run_projection(proj, X, y, id_run, dataset_name, output_dir):
    t0 = perf_counter()
    X_new = proj.fit_transform(X, y)
    elapsed_time = perf_counter() - t0

    proj_name = id_run.split('|')[0]

    # if proj_name in projections_to_save:
    #     clean_id_run = metrics.cleanup_id_run(id_run)
    #     joblib.dump(proj, os.path.join(output_dir, '%s_projection_model_%s.pkl' % (dataset_name, clean_id_run)))

    return X_new, y #, metrics.eval_pq_metrics(X=X_new, y=y, elapsed_time=elapsed_time, id_run=id_run, dataset_name=dataset_name, output_dir=output_dir)


all_projections['PCA'] = (decomposition.PCA(),
                          {'n_components': [2],
                           'random_state': [42]})

all_projections['FactorAnalysis'] = (decomposition.FactorAnalysis(),
                                     {'n_components': [2],
                                         'max_iter': [1000],
                                         'random_state': [42]})

all_projections['FastICA'] = (decomposition.FastICA(),
                              {'n_components': [2],
                               'fun': ['exp'],
                               'max_iter': [200],
                               'random_state': [42]})

all_projections['KernelPCASigmoid'] = (decomposition.KernelPCA(),
                                       {'n_components': [2],
                                        'gamma': [None],
                                        'degree': [3],
                                        'kernel': ['sigmoid'],
                                        'max_iter': [None],
                                        'random_state': [42]})

all_projections['KernelPCALinear'] = (decomposition.KernelPCA(),
                                      {'n_components': [2],
                                       'kernel': ['linear'],
                                       'max_iter': [None],
                                       'random_state': [42]})

all_projections['KernelPCAPoly'] = (decomposition.KernelPCA(),
                                    {'n_components': [2],
                                     'gamma': [None],
                                     'degree': [2],
                                     'kernel': ['poly'],
                                     'max_iter': [None],
                                     'random_state': [42]})

all_projections['KernelPCARbf'] = (decomposition.KernelPCA(),
                                   {'n_components': [2],
                                    'gamma': [None],
                                    'kernel': ['rbf'],
                                    'max_iter': [None],
                                    'random_state': [42]})

all_projections['SparsePCA'] = (decomposition.SparsePCA(),
                                {'n_components': [2],
                                 'max_iter': [1000],
                                 'tol': [1e-08],
                                 'method': ['lars'],
                                 'random_state': [42]})

all_projections['Isomap'] = (manifold.Isomap(),
                             {'n_components': [2],
                              'n_neighbors': [7],
                              'max_iter': [100]})

all_projections['LLE'] = (manifold.LocallyLinearEmbedding(),
                          {'n_components': [2],
                           'n_neighbors': [7],
                           'max_iter': [100],
                           'method': ['standard'],
                           'eigen_solver': ['dense'],
                           'random_state': [42]})

all_projections['HessianLLE'] = (manifold.LocallyLinearEmbedding(),
                                 {'n_components': [2],
                                  'n_neighbors': [7],
                                  'max_iter': [100],
                                  'method': ['hessian'],
                                  'eigen_solver': ['dense'],
                                  'random_state': [42]})

all_projections['MLLE'] = (manifold.LocallyLinearEmbedding(),
                           {'n_components': [2],
                            'n_neighbors': [7],
                            'max_iter': [100],
                            'method': ['modified'],
                            'eigen_solver': ['dense'],
                            'random_state': [42]})

all_projections['LTSA'] = (manifold.LocallyLinearEmbedding(),
                           {'n_components': [2],
                            'n_neighbors': [7],
                            'max_iter': [100],
                            'method': ['ltsa'],
                            'eigen_solver': ['dense'],
                            'random_state': [42]})

all_projections['MetricMDS'] = (manifold.MDS(),
                          {'n_components': [2],
                           'n_init': [4],
                           'metric': [True],
                           'max_iter': [300],
                           'random_state': [42]})


all_projections['NonMetricMDS'] = (manifold.MDS(),
                          {'n_components': [2],
                           'n_init': [4],
                           'metric': [False],
                           'max_iter': [300],
                           'random_state': [42]})

all_projections['LaplacianEigenmaps'] = (manifold.SpectralEmbedding(),
                                         {'n_components': [2],
                                          'affinity': ['nearest_neighbors'],
                                          'gamma': [None],
                                          'random_state': [42]})

#FIXME: only supports n_components < n_classes -1 - not suitable for binary problems
# all_projections['LinearDiscriminantAnalysis'] = (discriminant_analysis.LinearDiscriminantAnalysis(),
#                                                  {'n_components': [2]})

all_projections['SparseRandomProjection'] = (random_projection.SparseRandomProjection(),
                                             {'n_components': [2],
                                              'density': ['auto'],
                                              'random_state': [42]})

all_projections['GaussianRandomProjection'] = (random_projection.GaussianRandomProjection(),
                                               {'n_components': [2],
                                                'random_state': [42]})

all_projections['TSNE'] = (mtsne.MTSNE(),
                           {'n_components': [2],
                            'perplexity': [20.0],
                            'learning_rate': [200.0],
                            'n_iter': [3000],
                            'n_iter_without_progress': [300],
                            'min_grad_norm': [1e-07],
                            'metric': ['euclidean'],
                            'init': ['random'],
                            'random_state': [42],
                            'method': ['barnes_hut'],
                            'angle': [0.5],
                            'n_jobs': [4]})

all_projections['UMAP'] = (umap.UMAP(),
                           {'n_components': [2],
                            'random_state': [42],
                            'n_neighbors': [10],
                            'metric': ['euclidean']})

all_projections['LAMP'] = (vp.LAMP(),
                           {'command': [os.getcwd() + '/vispipeline/vp-run'],
                            'verbose': [False],
                            'fraction_delta': [8.0],
                            'n_iterations': [100],
                            'sample_type': ['random']})

# all_projections['LSP'] = (vp.LSP(),
#                           {'command': [os.getcwd() + '/vispipeline/vp-run'],
#                            'verbose': [False],
#                            'fraction_delta': [8.0],
#                            'n_iterations': [100],
#                            'n_neighbors': [8],
#                            'control_point_type': ['random'],
#                            'dissimilarity_type': ['euclidean']})

all_projections['PLMP'] = (vp.PLMP(),
                           {'command': [os.getcwd() + '/vispipeline/vp-run'],
                            'verbose': [False],
                            'fraction_delta': [8.0],
                            'n_iterations': [100],
                            'sample_type': ['random'],
                            'dissimilarity_type': ['euclidean']})

all_projections['PLSP'] = (vp.PLSP(),
                           {'command': [os.getcwd() + '/vispipeline/vp-run'],
                            'dissimilarity_type': ['euclidean'],
                            'verbose': [False],
                            'sample_type': ['clustering']})

all_projections['IDMAP'] = (vp.IDMAP(),
                            {'command': [os.getcwd() + '/vispipeline/vp-run'],
                             'verbose': [False],
                             'fraction_delta': [8.0],
                             'n_iterations': [100],
                             'init_type': ['random'],
                             'dissimilarity_type': ['euclidean']})

all_projections['Fastmap'] = (vp.Fastmap(),
                              {'command': [os.getcwd() + '/vispipeline/vp-run'],
                               'verbose': [False],
                               'dissimilarity_type': ['euclidean']})

all_projections['RapidSammon'] = (vp.RapidSammon(),
                                  {'command': [os.getcwd() + '/vispipeline/vp-run'],
                                   'verbose': [False],
                                   'dissimilarity_type': ['euclidean']})

all_projections['LandmarkIsomap'] = (vp.LandmarkIsomap(),
                                     {'command': [os.getcwd() + '/vispipeline/vp-run'],
                                         'verbose': [False],
                                         'n_neighbors': [8],
                                         'dissimilarity_type': ['euclidean']})

all_projections['ProjectionByClustering'] = (vp.ProjectionByClustering(),
                                             {'command': [os.getcwd() + '/vispipeline/vp-run'],
                                              'verbose': [False],
                                              'fraction_delta': [8.0],
                                              'n_iterations': [100],
                                              'init_type': ['random'],
                                              'dissimilarity_type': ['euclidean'],
                                              'cluster_factor': [4.5]})

# all_projections['DiffusionMaps'] = (sgtk.DiffusionMaps(),
#                                     {'n_components': [2],
#                                      't': [5],
#                                      'width': [5.0]})

# all_projections['StochasticProximityEmbedding'] = (sgtk.StochasticProximityEmbedding(),
#                                                    {'n_components': [2],
#                                                     'n_neighbors': [7],
#                                                     'n_updates': [20],
#                                                     'max_iter': [0]})

# all_projections['KernelLocallyLinearEmbedding'] = (sgtk.KernelLocallyLinearEmbedding(),
#                                                    {'n_components': [2],
#                                                     'n_neighbors': [7]})

# all_projections['LocalityPreservingProjections'] = (sgtk.LocalityPreservingProjections(),
#                                                     {'n_components': [2],
#                                                      'n_neighbors': [7]})

# all_projections['LinearLocalTangentSpaceAlignment'] = (sgtk.LinearLocalTangentSpaceAlignment(),
#                                                        {'n_components': [2],
#                                                         'n_neighbors': [7]})

# all_projections['NeighborhoodPreservingEmbedding'] = (sgtk.NeighborhoodPreservingEmbedding(),
#                                                       {'n_components': [2],
#                                                        'n_neighbors': [7]})

# # TODO: not working, getting stuck
# # all_projections['ManifoldSculpting'] = (sgtk.ManifoldSculpting(),
# #                                         {'n_components': [2],
# #                                          'n_neighbors': [3, 5, 10],
# #                                          'squishing_rate': [0.1, 0.5, 0.8, 0.99],
# #                                          'max_iter': [20]})
