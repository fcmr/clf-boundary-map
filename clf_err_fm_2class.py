import numpy as np
import joblib
from boundarymap import CLF
from boundarymap import Grid
from boundarymap import PlotProjection
from boundarymap import PlotDenseMap

import matplotlib.pyplot as plt

from utils import *
from matplotlib.colors import hsv_to_rgb

from keras.models import load_model


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

    #colors[y_pred != y_true] = np.array([0.0, 0.0, 0.0, 0.8])

    # assumes that the projection is normalized to range [0.0, 1.0]
    #x_intrvls = np.linspace(0.0 - 1e-5, 1.0 + 1e-5, num=grid_size + 1)
    #y_intrvls = np.linspace(0.0 - 1e-5, 1.0 + 1e-5, num=grid_size + 1)
    #plt.axes().set_aspect('equal')
    #x_min, x_max = np.min(x_intrvls), np.max(x_intrvls)
    #y_min, y_max = np.min(y_intrvls), np.max(y_intrvls)
    #print(x_min, x_max)
    #print(y_min, y_max)
    #for x in x_intrvls:
    #    plt.plot([x,  x], [y_min, y_max], color='k')
    #for y in y_intrvls:
    #    plt.plot([x_min,  x_max], [y, y], color='k')

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


def PlotDenseMapErr(gsize, dense_map, proj_in, y_true, y_pred, title, filename):
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
    plt.scatter(gsize*proj[y_pred != y_true][:, 0], gsize*(1.0 - proj[y_pred != y_true][:, 1]), color=[1.0, 1.0, 0.0], s=3.0)
    #plt.scatter(100*proj[y_pred != y_true][:, 0], 100*(1.0 - proj[y_pred != y_true][:, 1]), color=colors, s=5.0)
    #plt.scatter(100*proj[y_pred != y_true][:, 0], 100*(1.0 - proj[y_pred != y_true][:, 1]), color=colors[y_pred != y_true], s=3.0, edgecolor=edge_colors[y_pred != y_true], linewidth=0.5)

    idx = np.arange(len(y_pred))
    miss_idx = idx[y_pred != y_true]
    for v in miss_idx:
        print("anottatting ", v)
        #plt.annotate(str(v), (gsize*proj[v, 0], gsize*(1.0 - proj[v, 1])))

    plt.title(title)

    plt.savefig(filename + ".pdf", format='pdf')
    #plt.show()
    plt.clf()
    #Image.fromarray((rgb_img*255).astype(np.uint8)).save(filename + "_arr.png")
    #plt.imsave(filename + "_plt.svg", (rgb_img*255).astype(np.uint8),format='svg')
    #plt.clf()


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
BASE_DIR = 'data/fashionmnist/'

projections_f = [
        'projections/fashionmnist_MetricMDS_True_projected.pkl',
        'projections/fashionmnist_PLMP_True_projected.pkl',
        'projections/fashionmnist_ProjectionByClustering_True_projected.pkl',
        'projections/fashionmnist_TSNE_True_projected.pkl',
        'projections/fashionmnist_UMAP_True_projected.pkl' ]

clfs_f = [
        'clfs/fashionmnist_model_KNeighborsClassifier_True.pkl',
        'clfs/fashionmnist_model_Sequential_True.h5']

dmaps_f = [
        'dmaps/fashionmnist_DenseMap_400x400_N_5_MetricMDS_KNeighborsClassifier.npy',
        'dmaps/fashionmnist_DenseMap_400x400_N_5_MetricMDS_Sequential.npy',

        'dmaps/fashionmnist_DenseMap_400x400_N_5_PLMP_KNeighborsClassifier.npy',
        'dmaps/fashionmnist_DenseMap_400x400_N_5_PLMP_Sequential.npy',

        'dmaps/fashionmnist_DenseMap_400x400_N_5_ProjectionByClustering_KNeighborsClassifier.npy',
        'dmaps/fashionmnist_DenseMap_400x400_N_5_ProjectionByClustering_Sequential.npy',
        
        'dmaps/fashionmnist_DenseMap_400x400_N_5_TSNE_KNeighborsClassifier.npy',
        'dmaps/fashionmnist_DenseMap_400x400_N_5_TSNE_Sequential.npy',

        'dmaps/fashionmnist_DenseMap_400x400_N_5_UMAP_KNeighborsClassifier.npy',
        'dmaps/fashionmnist_DenseMap_400x400_N_5_UMAP_Sequential.npy'
        ]

P_MMDS = 0
P_PLMP = 1
P_PBC = 2
P_TSNE = 3
P_UMAP = 4

CLF_KNN = 0
CLF_CNN = 1

DMAP_MMDS_KNN = 0
DMAP_MMDS_CNN = 1
DMAP_PLMP_KNN = 2
DMAP_PLMP_CNN = 3
DMAP_PBC_KNN = 4
DMAP_PBC_CNN = 5
DMAP_TSNE_KNN = 6
DMAP_TSNE_CNN = 7
DMAP_UMAP_KNN = 8
DMAP_UMAP_CNN = 9

GRID_SIZE = 400

# Load the dataset
print("Loading dataset")
X_train = np.load(BASE_DIR + 'X_sample_bin.npy')
y_train = np.load('data/fashionmnist/y_sample_bin.npy')
# TODO: y_train contains the values 0 and 9, replace all 9s for 1s?
y_train[y_train == 9] = 1


print("Loading projections")
projs = [LoadProjection(BASE_DIR + f) for f in projections_f]

print("Loading classifiers")
clfs = [
        CLF(clf=joblib.load(open(BASE_DIR + clfs_f[0], 'rb')), clf_type="sklearn"),
        CLF(clf=load_model(BASE_DIR + clfs_f[1]), clf_type="keras_cnn", shape=(28, 28, 1))]

print("Loading densemaps")
dmaps = [np.load(BASE_DIR + f) for f in dmaps_f]


print("Predicting labels for training set")
y_preds = [clf.Predict(X_train) for clf in clfs]

print("PLotting densemap errors...")
# MMDS KNN
PlotDenseMapErr(GRID_SIZE, dmaps[0], projs[P_MMDS], y_train, y_preds[CLF_KNN], "ERR DENSEMAP MMDS KNN", BASE_DIR + 'err_densemap_MMDS_KNN')
# MMDS CNN 
PlotDenseMapErr(GRID_SIZE, dmaps[1], projs[P_MMDS], y_train, y_preds[CLF_CNN], "ERR DENSEMAP MMDS CNN", BASE_DIR + 'err_densemap_MMDS_CNN')

# PLMP KNN
PlotDenseMapErr(GRID_SIZE, dmaps[2], projs[P_PLMP], y_train, y_preds[CLF_KNN], "ERR DENSEMAP PLMP KNN", BASE_DIR + 'err_densemap_PLMP_KNN')
# PLMP CNN 
PlotDenseMapErr(GRID_SIZE, dmaps[3], projs[P_PLMP], y_train, y_preds[CLF_CNN], "ERR DENSEMAP PLMP CNN", BASE_DIR + 'err_densemap_PLMP_CNN')

# PBC KNN
PlotDenseMapErr(GRID_SIZE, dmaps[4], projs[P_PBC], y_train, y_preds[CLF_KNN], "ERR DENSEMAP Proj By Cluster KNN", BASE_DIR + 'err_densemap_PBC_KNN')
# PBC CNN 
PlotDenseMapErr(GRID_SIZE, dmaps[5], projs[P_PBC], y_train, y_preds[CLF_CNN], "ERR DENSEMAP Proj By Cluster CNN", BASE_DIR + 'err_densemap_PBC_CNN')

# TSNE KNN
PlotDenseMapErr(GRID_SIZE, dmaps[6], projs[P_TSNE], y_train, y_preds[CLF_KNN], "ERR DENSEMAP TSNE KNN", BASE_DIR + 'err_densemap_TSNE_KNN')
# TSNE CNN 
PlotDenseMapErr(GRID_SIZE, dmaps[7], projs[P_TSNE], y_train, y_preds[CLF_CNN], "ERR DENSEMAP TSNE CNN", BASE_DIR + 'err_densemap_TSNE_CNN')

# UMAP KNN
PlotDenseMapErr(GRID_SIZE, dmaps[8], projs[P_UMAP], y_train, y_preds[CLF_KNN], "ERR DENSEMAP UMAP KNN", BASE_DIR + 'err_densemap_UMAP_KNN')
# UMAP CNN 
PlotDenseMapErr(GRID_SIZE, dmaps[9], projs[P_UMAP], y_train, y_preds[CLF_CNN], "ERR DENSEMAP UMAP CNN", BASE_DIR + 'err_densemap_UMAP_CNN')



#################
#
##proj_file = 'fashionmnist_LAMP_True_projected.pkl'
#proj_file = 'fashionmnist_UMAP_True_projected.pkl'
#proj_dict = joblib.load(open(base_dir+ proj_file, 'rb'))
#key = next(iter(proj_dict))
#proj = proj_dict[key]['X']
## note: proj_dict[key]['y'] == y_train, will read it in another file
##proj_y = proj_dict[key]['y']
#
##X_train = np.load(base_dir + 'X_sample_bin.npy')
#X_train = np.load(base_dir + 'X_sample_bin.npy')
#y_train = np.load('data/fashionmnist/y_sample_bin.npy')
## TODO: y_train contains the values 0 and 9, replace all 9s for 1s?
#y_train[y_train == 9] = 1
#
##clf_file = 'fashionmnist_model_LogisticRegression_True.pkl'
##clf_sk = joblib.load(open(base_dir + clf_file, 'rb'))
##clf_logreg = CLF(clf=clf_sk, clf_type="sklearn")
##y_pred = clf_logreg.Predict(X_train)
#
#clf_file = 'fashionmnist_model_Sequential_True.h5'
#clf_cnn = CLF()
#input_shape = (28, 28, 1)
#clf_cnn.LoadKerasModel(base_dir + clf_file, name="CNN", shape=input_shape)
#y_pred = clf_cnn.Predict(X_train)
#
## X_test ?
## y_test ?
#
##PlotProjectionErr(dmap.shape[0], proj, clf_logreg.Predict(X_train), y_train, base_dir, 'teste err') 
#PlotProjectionErr(dmap.shape[0], proj, clf_cnn.Predict(X_train), y_train, base_dir + 'proj_err_umap', 'teste err') 
#
#plot_path = base_dir + 'projection_cnn_umap.pdf'
#leg_path = base_dir + 'projection_legend.pdf'
#labels = ['0', '9']
#PlotProjection(proj, y_pred, plot_path, 'test', leg_path, labels)
#PlotDenseMap(dmap, 'test', base_dir + 'densemap_cnn_umap')
#
#PlotDenseMapErr(dmap, proj, y_train, y_pred, "densemap err", base_dir + 'err_densemap_cnn_umap')
#
## find the misclassified samples
#idx = np.arange(len(y_train))
#miss_idx = idx[y_train != y_pred]
#print(miss_idx)
#
#plt.clf()
#x = X_train[2226]
#plt.clf()
#plt.imshow(np.reshape(x, (28,28)), cmap='gray')
#plt.savefig(base_dir + 'x_2226.png', format='png')
#x = X_train[1297]
#plt.clf()
#plt.imshow(np.reshape(x, (28,28)), cmap='gray')
#plt.savefig(base_dir + 'x_1297.png', format='png')
#
#
## find the cell in where the misclassified sample is
#proj_miss = proj[y_train != y_pred]
#grid = Grid(proj_miss, 100)
#
#
