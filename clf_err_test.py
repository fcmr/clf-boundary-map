import numpy as np
np.random.seed(0)

import json
from boundarymap import CLF
from boundarymap import Grid 
from boundarymap import PlotDenseMap 
from boundarymap import PlotProjection

import matplotlib.pyplot as plt


from utils import *
from matplotlib.colors import hsv_to_rgb

def PlotProjectionErr(grid, X, y_pred, y_true, path, title, leg_path="", labels=[]):
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

    #colors[y_pred != y_true] = np.array([0.0, 0.0, 0.0, 0.8])

    plt.axes().set_aspect('equal')
    x_min, x_max = np.min(grid.x_intrvls), np.max(grid.x_intrvls)
    y_min, y_max = np.min(grid.y_intrvls), np.max(grid.y_intrvls)
    print(x_min, x_max)
    print(y_min, y_max)
    for x in grid.x_intrvls:
        plt.plot([x,  x], [y_min, y_max], color='k')
    for y in grid.y_intrvls:
        plt.plot([x_min,  x_max], [y, y], color='k')

    plt.scatter(X[y_pred == y_true][:, 0], X[y_pred == y_true][:, 1], color=colors[y_pred == y_true], s=10.0)
    plt.scatter(X[y_pred != y_true][:, 0], X[y_pred != y_true][:, 1], color=[0.0, 0.0, 0.0, 0.8], s=10.0, marker='*')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    print(plt.axes().get_xlim())
    print(plt.axes().get_ylim())
    #plt.savefig(path, format='pdf')

    plt.show()
    plt.clf()

    #if leg_path != "":
    #    PlotLegend(leg_path, COLORS[:len(labels)], labels)


def PlotDenseMapErr(grid, dense_map, proj_in, y_true, y_pred, title, filename):
    proj = np.copy(proj_in)
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


    tmp_dense = np.flip(dense_map, axis=0)
    tmp_dense = TransferFunc(tmp_dense, 0.7)
    rgb_img = hsv_to_rgb(tmp_dense)

    plt.xticks([])
    plt.yticks([])

    plt.axes().set_aspect('equal')
    plt.imshow(rgb_img, interpolation='none')
    cell_len = grid.x_intrvls[1] - grid.x_intrvls[0]
    proj -= np.array([cell_len*0.5, -cell_len*0.5])
    #x_min, x_max = np.min(grid.x_intrvls), np.max(grid.x_intrvls)
    #y_min, y_max = np.min(grid.y_intrvls), np.max(grid.y_intrvls)
    #x_min -= cell_len*0.5
    #x_max -= cell_len*0.5
    #y_min += cell_len*0.5
    #y_max += cell_len*0.5
    #print(x_min, x_max)
    #print(y_min, y_max)
    #for x in grid.x_intrvls:
    #    x -= cell_len*0.5
    #    plt.plot([50*x, 50*x], [50*(1.0 - y_min), 50*(1.0 - y_max)], color='k')
    #for y in grid.y_intrvls:
    #    y += cell_len*0.5
    #    plt.plot([50*x_min, 50*x_max], [50*(1.0 - y), 50*(1.0 - y)], color='k')

    #plt.scatter(50*proj[y_pred == y_true][:, 0], 50*(1.0 - proj[y_pred == y_true][:, 1]), color=colors[y_pred == y_true], s=4.0)
    plt.scatter(50*proj[y_pred != y_true][:, 0], 50*(1.0 - proj[y_pred != y_true][:, 1]), color=[1.0, 1.0, 1.0, 0.7], s=10.0)

    plt.title(title)

    #plt.savefig(filename + ".pdf", format='pdf')
    plt.show()
    #plt.clf()
    #Image.fromarray((rgb_img*255).astype(np.uint8)).save(filename + "_arr.png")
    #plt.imsave(filename + "_plt.svg", (rgb_img*255).astype(np.uint8),format='svg')
    #plt.clf()
 

def CalcErrMap(grid, dense_map, X, y_true, clf):
    #tmp_dense = np.flip(dense_map, axis=0)
    tmp_dense = TransferFunc(dense_map, 0.7)
    dmap = hsv_to_rgb(tmp_dense)

    grid_size = grid.grid_size
    errmap = np.zeros((grid_size*3, grid_size*3, 3))
    # make every pixel fully transparent

    for row in range(grid_size):
        for col in range(grid_size):
            errmap[3*row:3*(row + 1), 3*col:3*(col + 1)] = dmap[row, col]

            num_pts = len(grid.cells[row][col])
            if num_pts == 0:
                continue

            #X_sub = [x for x in X[grid.cells[row][col]]]
            X_sub = X[grid.cells[row][col]]
            y_sub = y_true[grid.cells[row][col]]
            y_pred = clf.Predict(X_sub)
            miss = (y_pred != y_sub).any()
            if not miss:
                continue
            # there was a miss, set color to black
            errmap[3*row + 1, 3*col + 1] = np.array([0.0, 0.0, 0.0])
    return errmap

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

    R = 50
    N = 1
    grid_logreg = Grid(proj, R)
    PlotProjectionErr(grid_logreg, proj, y_pred, y_train, path, title)
    print("Create grid logreg")
    _, dmap = grid_logreg.BoundaryMap(X_train, N, clf_logreg)

    fig_title = "{}x{} DenseMap Err ({} samples, {})".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    fig_name = "data/segmentation/DenseMap_Err_{}x{}_N_{}_dense_map_{}".format(grid_logreg.grid_size, grid_logreg.grid_size, N, clf_logreg.name)
    miss_classified = proj[y_train != y_pred]
    PlotDenseMapErr(grid_logreg, dmap, proj, y_train, y_pred, fig_title, fig_name)

    errmap = CalcErrMap(grid_logreg, dmap, X_train, y_train, clf_logreg) 
    tmp_err = np.flip(errmap, axis=0)
    plt.xticks([])
    plt.yticks([])
    #tmp_dense = np.flip(dmap, axis=0)
    #tmp_dense = TransferFunc(tmp_dense, 0.7)
    #rgb_img = hsv_to_rgb(tmp_dense)
    #plt.imshow(rgb_img, interpolation='none')
    plt.imshow(tmp_err, interpolation='none')
    plt.show()


if __name__ == "__main__":
    main()
