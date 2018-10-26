import numpy as np
np.random.seed(0)

import json
from boundarymap import CLF
from boundarymap import Grid 
from boundarymap import PlotDenseMap 
from boundarymap import PlotProjection

def main():
    # Ideal path:
    # 1 - Load dataset, projection and a trained classifier
    with open("data/segmentation/seg.json") as f:
        data_json = json.load(f)

    proj = np.load(data_json["proj"])

    X_train = np.load(data_json['X_train'])
    y_train = np.load(data_json['y_train'])
    X_test = np.load(data_json['X_test'])
    y_test = np.load(data_json['y_test'])

    clf_logreg = CLF()
    clf_logreg.LoadSKLearn(data_json['clfs'][0], "Logistic Regression")

    # Plots the projected points coulored according to the label assigned by
    # the classifier.
    # As it is the first projection plotted, the legend is also save into a 
    # separate file
    y_pred = clf_logreg.Predict(X_train)
    labels = ["0", "1", "2", "3", "4", "5", "6"]
    path = "data/segmentation/projection_logreg.pdf"
    path_leg = "data/segmentation/projection_leg.pdf"
    title = "LAMP Projection (Logistic Regression)"
    PlotProjection(proj, y_pred, path, title, path_leg, labels)

    clf_svm = CLF()
    clf_svm.LoadSKLearn(data_json['clfs'][1], "SVM")
    y_pred = clf_svm.Predict(X_train)
    path = "data/segmentation/projection_svm.pdf"
    title = "LAMP Projection (SVM)"
    PlotProjection(proj, y_pred, path, title)

    clf_knn5 = CLF()
    clf_knn5.LoadSKLearn(data_json['clfs'][2], "KNN (5)")
    y_pred = clf_knn5.Predict(X_train)
    path = "data/segmentation/projection_knn5.pdf"
    title = "LAMP Projection (KNN)"
    PlotProjection(proj, y_pred, path, title)


    # 2 - Run boundary map construction function on clf_logreg
    R = 500
    N = 5
    grid_logreg = Grid(proj, R)
    print("Create grid logreg")
    _, dmap = grid_logreg.BoundaryMap(X_train, N, clf_logreg)

    fig_title = "{}x{} DenseMap ({} samples, {})".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    fig_name = "data/segmentation/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    PlotDenseMap(dmap, fig_title, fig_name)

    # Run boundary map construction function on clf_svm
    grid_svm = Grid(proj, R)
    print("Create grid svm")
    _, dmap = grid_svm.BoundaryMap(X_train, N, clf_svm)

    fig_title = "{}x{} DenseMap ({} samples, {})".format(grid_svm.grid_size, grid_svm.grid_size, N, clf_svm.name)
    fig_name = "data/segmentation/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid_svm.grid_size, grid_svm.grid_size, N, clf_svm.name)
    PlotDenseMap(dmap, fig_title, fig_name)

    # Run boundary map construction function on clf_knn5
    grid_knn5 = Grid(proj, R)
    print("Create grid knn")
    _, dmap = grid_knn5.BoundaryMap(X_train, N, clf_knn5)

    fig_title = "{}x{} DenseMap ({} samples, {})".format(grid_knn5.grid_size, grid_knn5.grid_size, N, clf_knn5.name)
    fig_name = "data/segmentation/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid_knn5.grid_size, grid_knn5.grid_size, N, clf_knn5.name)
    PlotDenseMap(dmap, fig_title, fig_name)

    


if __name__ == "__main__":
    main()
