import numpy as np
np.random.seed(0)

import json
from boundarymap import CLF
from boundarymap import Grid 
from boundarymap import PlotDenseMap 
from boundarymap import PlotProjection

def PlotMatrix(proj, X, clf):
    print("PlotMatrix")
    for R in [50, 100, 200, 500]:
        grid = Grid(proj, R)
        for num_samples in [1, 5, 10, 15]:
            print("R:{} N:{}".format(R, num_samples))
            _, dmap = grid.BoundaryMap(X, num_samples, clf)
            fig_title = "{}x{} DenseMap ({} samples, {})".format(grid.grid_size, grid.grid_size, num_samples, clf.name)
            fig_name = "data/toy/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid.grid_size, grid.grid_size, num_samples, clf.name)
            PlotDenseMap(dmap, fig_title, fig_name)
    print("Finished PlotMatrix")

def main():
    # Ideal path:
    # 1 - Load dataset, projection and a trained classifier
    with open("data/toy/toy.json") as f:
        data_json = json.load(f)

    proj = np.load(data_json["proj"])

    X_train = np.load(data_json['X_train'])
    y_train = np.load(data_json['y_train'])
    X_test = np.load(data_json['X_test'])
    y_test = np.load(data_json['y_test'])

    clf = CLF()
    clf.LoadSKLearn(data_json['clfs'][0], "Logistic Regression")

    # Plots the projected points coulored according to the label assigned by
    # the classifier.
    # As it is the first projection plotted, the legend is also save into a 
    # separate file
    y_pred = clf.Predict(X_train)
    labels = ["0", "1"]
    path = "data/toy/projection_clf1_tsne.pdf"
    path_leg = "data/toy/projection_leg.pdf"
    title = "LAMP Projection"
    PlotProjection(proj, y_pred, path, title, path_leg, labels)

    #PlotMatrix(proj, X_train, clf)

    # 2 - Run boundary map construction function
    grid = Grid(proj, 100)
    num_samples = 5 
    _, dmap = grid.BoundaryMap(X_train, num_samples, clf)
    fig_title = "{}x{} DenseMap ({} samples, {})".format(grid.grid_size, grid.grid_size, num_samples, clf.name)
    fig_name = "data/toy/DenseMap_{}x{}_N_{}_dense_map_{}".format(grid.grid_size, grid.grid_size, num_samples, clf.name)
    PlotDenseMap(dmap, fig_title, fig_name)
    

if __name__ == "__main__":
    main()
