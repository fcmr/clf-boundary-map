import numpy as np
np.random.seed(0)

import json
from boundarymap import CLF
from boundarymap import Grid 
from boundarymap import PlotDenseMap 
from boundarymap import PlotProjection

import matplotlib.pyplot as plt

def PlotProjectionErr(X, y_pred, y_true, path, title, leg_path="", labels=[]):
    # COLORS are the rgb counterparts of Grid.CMAP_SYN
    COLORS = np.array([[0.09, 0.414, 0.9, 0.25],
                       [0.9, 0.333, 0.09, 0.25],
                       [0.09, 0.9, 0.171, 0.25],
                       [0.9, 0.09, 0.819, 0.25],
                       [0.495, 0.09, 0.9, 0.25],
                       [0.495, 0.9, 0.09, 0.25],
                       [0.09, 0.657, 0.9, 0.25],
                       [0.9, 0.09, 0.333, 0.25],
                       [0.9, 0.819, 0.09, 0.25],
                       [0.09, 0.9, 0.657, 0.25]])

    colors = [COLORS[i] for i in y_pred]
    colors = np.array(colors)

    colors[y_pred != y_true] = np.array([0.0, 0.0, 0.0, 0.8])

    plt.axes().set_aspect('equal')
    plt.scatter(X[:, 0], X[:, 1], color=colors, s=10.0)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, format='pdf')
    #plt.show()
    #plt.clf()

    #if leg_path != "":
    #    PlotLegend(leg_path, COLORS[:len(labels)], labels)

def main():
    # Will try to plot errors caused by classfier 
    # using segmentation dataset
    with open("data/segmentation/seg.json") as f:
        data_json = json.load(f)

    proj = np.load(data_json["proj"])

    X_train = np.load(data_json['X_train'])
    y_train = np.load(data_json['y_train'])
    X_test = np.load(data_json['X_test'])
    y_test = np.load(data_json['y_test'])

    # Logistic regression had the lowest accuracy, so it will be used in this
    # test
    clf_logreg = CLF()
    clf_logreg.LoadSKLearn(data_json['clfs'][0], "Logistic Regression")

    y_pred = clf_logreg.Predict(X_train)
    path = "data/segmentation/projection_logreg_err.pdf"
    title = "LAMP Projection Error (Logistic Regression)"
    PlotProjectionErr(proj, y_pred, y_train, path, title)

    #R = 50
    #N = 5
    #grid_logreg = Grid(proj, R)
    #print("Create grid logreg")
    #_, dmap = grid_logreg.BoundaryMap(X_train, N, clf_logreg)

    #fig_title = "{}x{} DenseMap Err ({} samples, {})".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    #fig_name = "data/segmentation/DenseMap_Err_{}x{}_N_{}_dense_map_{}".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    #PlotDenseMap(dmap, fig_title, fig_name)

if __name__ == "__main__":
    main()
