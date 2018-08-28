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
    with open("data/mnist/mnist.json") as f:
        data_json = json.load(f)

    proj_lamp = np.load(data_json["proj1"])
    proj_tsne = np.load(data_json["proj2"])

    X_train = np.load(data_json['X_train'])
    y_train = np.load(data_json['y_train'])
    X_test = np.load(data_json['X_test'])
    y_test = np.load(data_json['y_test'])

    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    clf1 = CLF()
    clf_path = data_json['clfs'][0]
    weights_path = data_json['weights'][0]
    clf1.LoadKeras(clf_path, weights_path, "CNN 1", input_shape)
    # Plots the projected points coulored according to the label assigned by
    # the classifier.
    # As it is the first projection plotted, the legend is also save into a 
    # separate file
    y_pred = clf1.Predict(X_train[:len(proj_tsne)])
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    path = "data/mnist/projection_clf1_tsne.pdf"
    path_leg = "data/mnist/projection_leg.pdf"
    title = "t-SNE Projection"
    PlotProjection(proj_tsne, y_pred, path, title, path_leg, labels)

    # Plots the LAMP projection for the same dataset.
    path = "data/mnist/projection_clf1_lamp.pdf"
    title = "LAMP Projection"
    PlotProjection(proj_lamp, y_pred, path, title)

    clf2_1 = CLF()
    clf_path = data_json['clfs'][1]
    weights_path = data_json['weights'][1]
    clf2_1.LoadKeras(clf_path, weights_path, "CNN 2 - 1 epoch", input_shape)

    clf2_5 = CLF()
    weights_path = data_json['weights'][2]
    clf2_5.LoadKeras(clf_path, weights_path, "CNN 2 - 5 epochs", input_shape)

    clf2_10 = CLF()
    weights_path = data_json['weights'][3]
    clf2_10.LoadKeras(clf_path, weights_path, "CNN 2 - 10 epochs", input_shape)

    clf2_50 = CLF()
    weights_path = data_json['weights'][4]
    clf2_50.LoadKeras(clf_path, weights_path, "CNN 2 - 50 epochs", input_shape)

     2 - Run boundary map construction function
    R = 300
    N = [1, 5, 10, 15]
    grid1 = Grid(proj_tsne, R)

    for n in N:
        print("Create densemap for ", n)
        _, dmap = grid1.BoundaryMap(X_train[:len(proj_tsne)], n, clf1)
        fig_title = "{}x{} DenseMap ({} samples, {})".format(grid1.grid_size, grid1.grid_size, n, clf1.name)
        fig_name = "data/mnist/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid1.grid_size, grid1.grid_size, n, clf1.name)
        PlotDenseMap(dmap, fig_title, fig_name)

    N = 1
    R = 300
    clf_epochs = [clf2_1, clf2_5, clf2_10, clf2_50]
    for clf in clf_epochs:
        print("Densemap for certain epoch")
        grid = Grid(proj_tsne, R)
        _, dmap = grid.BoundaryMap(X_train[:len(proj_tsne)], N, clf)
        fig_title = "{}x{} DenseMap ({} samples, {})".format(grid.grid_size, grid.grid_size, N, clf.name)
        fig_name = "data/mnist/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid.grid_size, grid.grid_size, N, clf.name)
        PlotDenseMap(dmap, fig_title, fig_name)

    print("Create densemap for LAMP, N = 15", )
    N = 15
    R = 300
    grid2 = Grid(proj_lamp, R)
    _, dmap = grid2.BoundaryMap(X_train[:len(proj_tsne)], N, clf1)
    fig_title = "{}x{} DenseMap LAMP Projection ({} samples, {})".format(grid2.grid_size, grid2.grid_size, N, clf1.name)
    fig_name = "data/mnist/DenseMap_{}x{}_N_{}_dense_map_{}_LAMP".format(grid2.grid_size, grid2.grid_size, N, clf1.name)
    PlotDenseMap(dmap, fig_title, fig_name)


if __name__ == "__main__":
    main()
