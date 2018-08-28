import numpy as np
import pickle

from utils import *
import lamp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image

# check colorscheme.png for a visual explanation
def SV(c, d):
    if d <= 0.5:
        # a: dark red rgb(0.372, 0.098, 0.145) - hsv(0.972, 1.0, 0.5)
        Sa = 1.0
        Va = 0.5
        # b: dark gray rgb(0.2, 0.2, 0.2) - hsv(0.0, 0.0, 0.2)
        Sb = 0.0
        Vb = 0.2
        # c: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
        Sc = 0.0
        Vc = 0.5
        # d: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
        Sd = 1.0
        Vd = 1.0
        S = Lerp(Lerp(Sb, Sa, c), Lerp(Sc, Sd, c), 2.0*d)
        V = Lerp(Lerp(Vb, Va, c), Lerp(Vc, Vd, c), 2.0*d)
    else:
        # a: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
        Sa = 1.0
        Va = 1.0
        # b: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
        Sb = 0.0
        Vb = 0.5
        # c: light gray rgb(0.8, 0.8, 0.8) - hsv(0.0, 0.0, 0.8)
        Sc = 0.0
        Vc = 0.8
        # d: bright pink rgb(? , ?, ?) - hsv(0.0, 0.2, 1.0)
        Sd = 0.2
        Vd = 1.0
        S = Lerp(Lerp(Sb, Sa, c), Lerp(Sc, Sd, c), 2.0*d - 1.0)
        V = Lerp(Lerp(Vb, Va, c), Lerp(Vc, Vd, c), 2.0*d - 1.0)
    return S, V


# TODO: state pattern?
class CLF:
    def __init__(self, clf=None, clf_type="", shape=None):
        self.clf = clf
        
        if clf_type == "":
            self.clf_type = -1
        elif clf_type == "sklearn":
            self.clf_type = 0
        elif clf_type == "keras_cnn":
            self.clf_type = 1
        
        self.shape = shape

    def LoadSKLearn(self, path, name=""):
        self.clf_type = 0
        self.clf = pickle.load(open(path, "rb"))
        self.name = name

    def LoadKeras(self, arch_path, weight_path, name="", shape=None):
        from keras.models import model_from_json
        import keras
        self.clf_type = 1

        self.name = name
        self.shape = shape

        # Model reconstruction from JSON file
        with open(arch_path, 'r') as f:
            self.clf = model_from_json(f.read())
        self.clf.load_weights(weight_path)

    def Predict(self, X):
        if self.shape is not None:
            #print("X shape: ", X.shape)
            #print("clf.shape: ", self.shape)
            X_new = np.reshape(X, (X.shape[0],) + self.shape)
        else:
            X_new = X
        y_pred = self.clf.predict(X_new)

        if self.clf_type == 1:
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def Score(self, X, y):
        if self.clf_type == 0:
            return self.clf.score(X, y)

class Grid:
    def __init__(self, proj, grid_size):
        self.CMAP_ORIG = np.array([234, 0, 108, 288, 252, 72, 180, 324, 36, 144])/360.0
        self.CMAP_SYN = np.array([216, 18, 126, 306, 270, 90, 198, 342, 54, 162])/360.0

        self.grid_size = grid_size
        # cells will store the indices of the points that fall inside each cell
        # of the grid
        # Initializes cells
        self.cells = [[] for i in range(grid_size)]

        for i in range(grid_size):
            self.cells[i] = [[] for _ in range(grid_size)]
        
        tile_size = 1.0/grid_size
        # Adds point's indices to the corresponding cell
        for idx in range(len(proj)):
            p = proj[idx]
            row = int(abs(p[1] - 1e-5)/tile_size)
            col = int(abs(p[0] - 1e-5)/tile_size)
            self.cells[row][col].append(idx)

        # TODO: projection is normalized [0.0, 1.0], thus min and max are known
        xmin = np.min(proj[:, 0])
        xmax = np.max(proj[:, 0])
        ymin = np.min(proj[:, 1])
        ymax = np.max(proj[:, 1])
        self.x_intrvls = np.linspace(xmin - 1e-5, xmax + 1e-5, num=grid_size + 1)
        self.y_intrvls = np.linspace(ymin - 1e-5, ymax + 1e-5, num=grid_size + 1)

        self.proj = proj

    # compute the max and average number of points in the grid, assumes that
    # each cell will have at least num_per_cell points 
    def GetMaxAvgPts(self, num_per_cell):
        num_pts = np.zeros((self.grid_size, self.grid_size)) 
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                num_pts[row, col] = len(self.cells[row][col])
        num_pts[num_pts < num_per_cell] = num_per_cell 
        return np.max(num_pts), np.mean(num_pts)
    
    # X_in: numpy array of original data points
    # num_samples: number of samples to create
    # TODO: this function should receive as input the sampled list
    def GenNewSamples(self, num_samples, X_in, row, col):
        if num_samples <= 0:
            return []

        rshp = False
        if len(X_in.shape) > 2:
            # image data will have shape (n, x, y) or even (n, x, y, z)
            # new_dim will be equal  to x*y and x*y*z
            new_dim = np.prod(X_in.shape[1:])
            X = np.reshape(X_in, (X_in.shape[0], new_dim))
            orig_shape = X_in.shape[1:] 
            rshp = True
        else:
            X = X_in

        limits = [self.x_intrvls[col], self.y_intrvls[row], self.x_intrvls[col + 1], self.y_intrvls[row + 1]]
        sampled = SampleSquare(num_samples, limits)
        new_X = []

        for (x, y) in sampled:
            new_sample = lamp.ilamp(X, self.proj, np.array([x, y]))
            if rshp is True:
                new_sample = new_sample.reshape(orig_shape)
            new_X.append(new_sample)
        return new_X

    # TODO: make this code work when num_per_cell == 0 -> sparse map
    def BoundaryMap(self, X, num_per_cell, clf, H=0.05):
        smap = np.zeros((self.grid_size, self.grid_size, 3))
        dmap = np.zeros((self.grid_size , self.grid_size, 3))

        max_pts, avg_pts = self.GetMaxAvgPts(num_per_cell)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                num_pts = len(self.cells[row][col])
                # TODO: remove sparsemap?
                if num_pts != 0:
                    cmap = self.CMAP_ORIG
                    #smap[row, col] = CellColor(X[cells[row][col]], clf, 
                    #                           num_pts, max_pts, avg_pts, cmap, H)
                else:
                    cmap = self.CMAP_SYN
                    smap[row, col, 2] = 1.0

                X_sub = [x for x in X[self.cells[row][col]]]
                # number of synthetic samples that will be created
                num_samples = num_per_cell - num_pts
                new_samples = self.GenNewSamples(num_samples, X, row, col)
                X_sub.extend(new_samples)
                X_sub = np.array(X_sub)

                #TODO: fix num_pts value
                # numpts is used to make computation on the lines below
                if num_pts < num_per_cell:
                    num_pts = num_per_cell

                luminance = num_pts/float(max_pts)
                # Compute color for this cell
                labels = clf.Predict(X_sub)
                #labels = PredictCLF(X_sub, clf, )
                #labels = clf.predict(X)

                counts = np.bincount(labels)
                num_winning = np.max(counts)
                # decision
                hue = cmap[np.argmax(counts)]
                # cconfusion c
                c = num_winning/num_pts
                # density d
                #d = num_pts/max_pts
                d = min(H*num_pts/avg_pts, 1.0)

                s, v = SV(c, d)
                dmap[row, col] = np.array([hue, s, v])
        return smap, dmap

def PlotDenseMap(dense_map, title, filename):
    tmp_dense = np.flip(dense_map, axis=0)
    tmp_dense = TransferFunc(tmp_dense, 0.7)
    rgb_img = hsv_to_rgb(tmp_dense)

    plt.xticks([])
    plt.yticks([])

    plt.imshow(rgb_img, interpolation='none')
    plt.title(title)
    plt.savefig(filename + ".pdf", format='pdf')
    plt.clf()
    Image.fromarray((rgb_img*255).astype(np.uint8)).save(filename + "_arr.png")
    plt.imsave(filename + "_plt.svg", (rgb_img*255).astype(np.uint8),format='svg')
    plt.clf()
    
def PlotLegend(path, colors, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    handles = []
    for c in colors:
        handles.append(ax.scatter([], [], color=c))

    figlegend = plt.figure()
    figlegend.legend(handles, labels, 'center')
    figlegend.savefig(path, format='pdf')
    plt.clf()

def PlotProjection(X, y_pred, path, title, leg_path="", labels=[]):
    # COLORS are the rgb counterparts of Grid.CMAP_SYN
    COLORS = np.array([[0.09, 0.414, 0.9, 0.5],
                       [0.9, 0.333, 0.09, 0.5],
                       [0.09, 0.9, 0.171, 0.5],
                       [0.9, 0.09, 0.819, 0.5],
                       [0.495, 0.09, 0.9, 0.5],
                       [0.495, 0.9, 0.09, 0.5],
                       [0.09, 0.657, 0.9, 0.5],
                       [0.9, 0.09, 0.333, 0.5],
                       [0.9, 0.819, 0.09, 0.5],
                       [0.09, 0.9, 0.657, 0.5]])

    colors = [COLORS[i] for i in y_pred]

    plt.axes().set_aspect('equal')
    plt.scatter(X[:, 0], X[:, 1], color=colors, s=10.0)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, format='pdf')
    plt.clf()

    if leg_path != "":
        PlotLegend(leg_path, COLORS[:len(labels)], labels)
