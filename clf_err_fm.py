import numpy as np
import joblib
from boundarymap import CLF
from boundarymap import Grid
from boundarymap import PlotProjection
from boundarymap import PlotDenseMap

import matplotlib.pyplot as plt

from utils import *
from matplotlib.colors import hsv_to_rgb

from pathlib import Path


def PlotProjectionErr(grid_size, proj, y_pred, y_true, path, title):
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

    edge_colors = [COLORS[i] for i in y_true]

    plt.scatter(proj[y_pred == y_true][:, 0], proj[y_pred == y_true][:, 1], color=colors[y_pred == y_true], s=10.0)
    plt.scatter(proj[y_pred != y_true][:, 0], proj[y_pred != y_true][:, 1], color=[0.0, 0.0, 0.0, 0.8], s=10.0, marker='*')
    num_errs = np.sum(y_pred != y_true)
    plt.title(title + "num errs: {}".format(num_errs))
    plt.xticks([])
    plt.yticks([])
    print(plt.axes().get_xlim())
    print(plt.axes().get_ylim())
    plt.savefig(path, format='pdf')

    #plt.show()
    plt.clf()


def PlotDenseMapErrOld(gsize, dense_map, proj_in, y_true, y_pred, title, filename):
    proj = np.copy(proj_in)

    COLORS = np.array([[0.09, 0.414, 0.9],
                       [0.9, 0.333, 0.09],
                       [0.09, 0.9, 0.171],
                       [0.9, 0.09, 0.819],
                       [0.495, 0.09, 0.9],
                       [0.495, 0.9, 0.09],
                       [0.09, 0.657, 0.9],
                       [0.9, 0.09, 0.333],
                       [0.9, 0.819, 0.09],
                       [0.09, 0.9, 0.657]])

    colors = [COLORS[i] for i in y_true]
    colors = np.array(colors)

    edge_colors = [COLORS[i] for i in y_pred]
    edge_colors = np.array(edge_colors)


    tmp_dense = np.flip(dense_map, axis=0)
    tmp_dense = TransferFunc(tmp_dense, 0.7)
    rgb_img = hsv_to_rgb(tmp_dense)

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0.0, gsize+0.5])
    plt.ylim([0.0, gsize+0.5])

    plt.axes().set_aspect('equal')
    plt.imshow(rgb_img, interpolation='none')
    #cell_len = grid.x_intrvls[1] - grid.x_intrvls[0]
    #proj -= np.array([cell_len*0.5, -cell_len*0.5])

    #plt.scatter(50*proj[y_pred == y_true][:, 0], 50*(1.0 - proj[y_pred == y_true][:, 1]), color=colors[y_pred == y_true], s=10.0)
    #plt.scatter(100*proj[y_pred != y_true][:, 0], 100*(1.0 - proj[y_pred != y_true][:, 1]), color=colors, s=5.0)
    #plt.scatter(100*proj[y_pred != y_true][:, 0], 100*(1.0 - proj[y_pred != y_true][:, 1]), color=colors[y_pred != y_true], s=3.0, edgecolor=edge_colors[y_pred != y_true], linewidth=0.5)

    plt.scatter(gsize*proj[:, 0], gsize*(1.0 - proj[:, 1]), color=[1.0, 1.0, 1.0, 0.3], s=5.0, linewidth=0.0)
    #idx = np.arange(len(y_pred))
    #miss_idx = idx[y_pred != y_true]
    #for v in miss_idx:
    #    print("anottatting ", v)
    #    plt.annotate(str(v), (gsize*proj[v, 0], gsize*(1.0 - proj[v, 1])))

    plt.title(title)

    plt.savefig(filename + ".pdf", format='pdf')
    #plt.show()
    plt.clf()
    #Image.fromarray((rgb_img*255).astype(np.uint8)).save(filename + "_arr.png")
    #plt.imsave(filename + "_plt.svg", (rgb_img*255).astype(np.uint8),format='svg')
    #plt.clf()

def PlotDenseMapErr(gsize, dense_map, proj_in, y_pred, title, filename):
    proj = np.copy(proj_in)
    tmp_dense = np.flip(dense_map, axis=0)
    tmp_dense = TransferFunc(tmp_dense, 0.7)
    rgb_img = hsv_to_rgb(tmp_dense)

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0.0, gsize+0.5])
    plt.ylim([0.0, gsize+0.5])

    plt.axes().set_aspect('equal')
    plt.imshow(rgb_img, interpolation='none')

    plt.scatter(gsize*proj[:, 0], gsize*(1.0 - proj[:, 1]), color=[1.0, 1.0, 1.0, 0.5], s=5.0, linewidth=0.0)

    plt.title(title)
    plt.savefig(filename + ".pdf", format='pdf')
    plt.clf()

# FIXME: proj is only used to compute the grid
def CheckMissClf(grid_size, proj, y_pred, y_true):
    grid = Grid(proj, grid_size)
    
    # data indices
    idx = np.arange(len(y_true))
    # misclassified indices
    miss_clfs = idx[y_true != y_pred]
    
    misclf_effective = []
    for row in range(grid.grid_size):
        for col in range(grid.grid_size):
            num_pts = len(grid.cells[row][col])
            if num_pts == 0:
                continue
            # turn it into a ndarray to allow comparison with == operator
            pts_in_cell = np.array(grid.cells[row][col])
            # check the indices of the misclassified points in this cell
            miss_clf_in_cell = pts_in_cell[np.in1d(pts_in_cell, miss_clfs)]
            if miss_clf_in_cell.size == 0:
                continue
            # at least one point in this cell is miss_clf, now check if they
            # affect cell color
            # check which class has the most points, that is, which class
            # defined this cell's color
            counts = np.bincount(y_pred[pts_in_cell])
            winning_label = np.argmax(counts)
            
            # check if the winning_label is among the misclassified
            # if it is, then a misclassified point affect this cell's color
            miss_clf_labels = y_pred[miss_clf_in_cell] 
            if winning_label in miss_clf_labels:
                miss_list = miss_clf_in_cell[miss_clf_labels == winning_label].tolist()
                misclf_effective = misclf_effective + miss_list
    return misclf_effective



def main():
    dmap_path = 'data/'
    #np.load()

if __name__ == "__main__":
    main()

def LoadProjection(path):
    proj_dict = joblib.load(open(path, 'rb'))
    key = next(iter(proj_dict))
    proj = proj_dict[key]['X']

    return proj

# temp main
GRID_SIZE = 400
BASE_DIR = 'data/fashionmnist_full/'
PROJS = ['MetricMDS', 'PLMP', 'ProjectionByClustering', 'TSNE', 'UMAP']
CLFS = ['KNeighborsClassifier', 'Sequential', 'LogisticRegression', 'RandomForestClassifier']

pred = BASE_DIR + 'projections/fashionmnist_'
suc = '_False_projected.pkl'
projections_f = [pred + p + suc for p in PROJS]

pred = BASE_DIR + 'clfs/fashionmnist_model_'
suc = '_False.pkl'
clfs_f = [pred + c + suc for c in CLFS]

pred = BASE_DIR + 'dmaps/fashionmnist_DenseMap_400x400_N_5_'
suc = '.npy'
dmaps_f = [pred + p + '_' + c + suc for p in PROJS for c in CLFS]


# Load the dataset
print("Loading dataset")
X_train = np.load(BASE_DIR + 'X_sample.npy')
y_train = np.load(BASE_DIR + 'y_sample.npy')

print("Checking if predictions exist, loading CLF and predicting otherwise")
y_preds_f = [BASE_DIR + 'y_sample_pred_' + c + '.npy' for c in CLFS]
y_pred_exists = [Path(y).is_file() for y in y_preds_f]

y_preds = [None]*len(CLFS)
for i in range(4):
    if y_pred_exists[i] is False:
        print("Needed to load classifier ", clfs_f[i])
        if i == 1:
            from keras import load_model
            clf = CLF(clf=load_model(clfs_f[i]), clf_type="keras_cnn", shape=(28, 28, 1))
        else:
            clf = CLF(clf=joblib.load(open(clfs_f[i], 'rb')), clf_type="sklearn")
        y_preds[i] = clf.Predict(X_train)
        np.save(y_preds_f[i], y_preds[i])
    else:
        y_preds[i] = np.load(y_preds_f[i])

print("Loading projections")
projs = [LoadProjection(f) for f in projections_f]

print("Loading densemaps")
dmaps = [np.load(f) for f in dmaps_f]

print("PLotting densemap errors...")
counter = 0
titles = ["MDS (Metric)", "PLMP", "Projection By Clustering", "t-SNE", "UMAP"]
for i in range(len(PROJS)):
    for j in range(len(CLFS)):
        #title = "{}".format(PROJS[i])
        path = BASE_DIR + 'err_densemap_{}_{}'.format(PROJS[i], CLFS[j])
        proj_in = projs[i][y_train != y_preds[j]]
        PlotDenseMapErr(GRID_SIZE, dmaps[counter], proj_in, y_train, y_preds[j], titles[i], path)
        
        path_eff = BASE_DIR + 'eff_err_densemap_{}_{}'.format(PROJS[i], CLFS[j])
        miss_eff = np.array(CheckMissClf(GRID_SIZE, projs[i], y_preds[j], y_train))
        proj_in_eff = projs[i][miss_eff]
        PlotDenseMapErr(GRID_SIZE, dmaps[counter], proj_in_eff, y_train, y_preds[j], titles[i], path_eff)


        counter += 1
