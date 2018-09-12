import argparse
import os
from glob import glob
import joblib
import numpy as np

import classifier
from boundarymap import Grid, PlotDenseMap, PlotProjection

import tensorflow as tf
import keras.backend as K
import random as rn
from sklearn.model_selection import train_test_split

def create_densemap(dataset_name, output_dir, model_name, projection_name, is_binary, grid_size=500, num_per_cell=5):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # begin set seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)

    tf.set_random_seed(42)

    session_conf = tf.ConfigProto()
    session_conf.intra_op_parallelism_threads = 1
    session_conf.inter_op_parallelism_threads = 1
    session_conf.gpu_options.allow_growth = True
    
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # end set seed

    print('Loading model %s' % model_name)
    clf = classifier.load_model(model_name)

    print('Loading dataset %s' % dataset_name)
    X, y = classifier.load_dataset(dataset_name, 'sample', is_binary, img=(clf.__class__.__name__ == 'Sequential'))

    print('Predicting...')
    y_pred = classifier.predict(clf, X)

    print('Getting labels...')
    labels = [str(i) for i in range(len(np.unique(y_pred)))]

    clf_name = os.path.basename(model_name)
    clf_name = clf_name.replace(dataset_name, '').replace('_model_', '').replace('_True.pkl', '').replace('_False.pkl', '').replace('_True.h5', '').replace('_False.h5', '')

    projected_data = joblib.load(projection_name)

    #FIXME: changed with the assumption of only one parameter set
    id_run = list(projected_data.keys())[0]

    print(clf_name, id_run)
    proj_name = id_run.split('|')[0]
    X_low = projected_data[id_run]['X']

    path = "%s/%s_%s_%s_%s_%d_%d.pdf" % (output_dir, dataset_name, clf_name, proj_name, str(is_binary), grid_size, num_per_cell)
    path_leg = "%s/%s_%s_%s_%s_%d_%d_leg.pdf"  % (output_dir, dataset_name, clf_name, proj_name, str(is_binary), grid_size, num_per_cell)
    title = "%s: %s (%s) binary=%s (%d, %d)" % (dataset_name, clf_name, proj_name, str(is_binary), grid_size, num_per_cell)
    
    print('Plotting projection: ', clf_name, proj_name)
    PlotProjection(X_low, y_pred, path, title, path_leg, labels)

    # 2 - Run boundary map construction function on clf_logreg
    R = grid_size
    N = num_per_cell
    grid = Grid(X_low, R)

    print('Creating boundary map: ', clf_name, id_run)
    _, dmap = grid.BoundaryMap(X, N, clf)

    fig_title = "{}: {}x{} DenseMap ({} samples, {})".format(dataset_name, grid.grid_size, grid.grid_size, N, clf_name)
    fig_name = "{}/{}_DenseMap_{}x{}_N_{}_{}_{}".format(output_dir, dataset_name, grid.grid_size, grid.grid_size, N, proj_name, clf_name)

    print('Plotting densemap: ', clf_name, proj_name)
    PlotDenseMap(dmap, fig_title, fig_name)


    # # Ideal path:
    # # 1 - Load dataset, projection and a trained classifier
    # with open("data/segmentation/seg.json") as f:
    #     data_json = json.load(f)

    # proj = np.load(data_json["proj"])

    # X_train = np.load(data_json['X_train'])
    # y_train = np.load(data_json['y_train'])
    # X_test = np.load(data_json['X_test'])
    # y_test = np.load(data_json['y_test'])

    # clf_logreg = CLF()
    # clf_logreg.LoadSKLearn(data_json['clfs'][0], "Logistic Regression")

    # # Plots the projected points coulored according to the label assigned by
    # # the classifier.
    # # As it is the first projection plotted, the legend is also save into a 
    # # separate file
    # y_pred = clf_logreg.Predict(X_train)
    # labels = ["0", "1", "2", "3", "4", "5", "6"]
    # path = "data/segmentation/projection_logreg.pdf"
    # path_leg = "data/segmentation/projection_leg.pdf"
    # title = "LAMP Projection (Logistic Regression)"
    # PlotProjection(proj, y_pred, path, title, path_leg, labels)

    # clf_svm = CLF()
    # clf_svm.LoadSKLearn(data_json['clfs'][1], "SVM")
    # y_pred = clf_svm.Predict(X_train)
    # path = "data/segmentation/projection_svm.pdf"
    # title = "LAMP Projection (SVM)"
    # PlotProjection(proj, y_pred, path, title)

    # clf_knn5 = CLF()
    # clf_knn5.LoadSKLearn(data_json['clfs'][2], "KNN (5)")
    # y_pred = clf_knn5.Predict(X_train)
    # path = "data/segmentation/projection_knn5.pdf"
    # title = "LAMP Projection (KNN)"
    # PlotProjection(proj, y_pred, path, title)


    # # 2 - Run boundary map construction function on clf_logreg
    # R = 500
    # N = 5
    # grid_logreg = Grid(proj, R)
    # print("Create grid logreg")
    # _, dmap = grid_logreg.BoundaryMap(X_train, N, clf_logreg)

    # fig_title = "{}x{} DenseMap ({} samples, {})".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    # fig_name = "data/segmentation/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    # PlotDenseMap(dmap, fig_title, fig_name)

    # # Run boundary map construction function on clf_svm
    # grid_svm = Grid(proj, R)
    # print("Create grid svm")
    # _, dmap = grid_svm.BoundaryMap(X_train, N, clf_svm)

    # fig_title = "{}x{} DenseMap ({} samples, {})".format(grid_svm.grid_size, grid_svm.grid_size, N, clf_svm.name)
    # fig_name = "data/segmentation/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid_svm.grid_size, grid_svm.grid_size, N, clf_svm.name)
    # PlotDenseMap(dmap, fig_title, fig_name)

    # # Run boundary map construction function on clf_knn5
    # grid_knn5 = Grid(proj, R)
    # print("Create grid knn")
    # _, dmap = grid_knn5.BoundaryMap(X_train, N, clf_knn5)

    # fig_title = "{}x{} DenseMap ({} samples, {})".format(grid_knn5.grid_size, grid_knn5.grid_size, N, clf_knn5.name)
    # fig_name = "data/segmentation/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid_knn5.grid_size, grid_knn5.grid_size, N, clf_knn5.name)
    # PlotDenseMap(dmap, fig_title, fig_name)
