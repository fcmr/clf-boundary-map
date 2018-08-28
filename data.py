import numpy as np
from sklearn import preprocessing 
import os
import struct
import sys

# TODO: train/test split?
def LoadZeroOneData():
    X = np.load("data/zero_one.npy")
    y = np.load("data/zero_one_label.npy")
    return X, y

def LoadWineData():
    data = np.loadtxt("data/wine/wine.data", delimiter=",")
    y = data[:, 0].astype(int) - 1
    X = data[:, 1:]
    # normalizes X
    X = preprocessing.MinMaxScaler().fit_transform(X)
    return X, y

def LoadSegmentationData():
    data = np.loadtxt("data/segmentation/segmentation.txt", delimiter=",", comments='#')
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    X = preprocessing.MinMaxScaler().fit_transform(X)
    return X, y


"""
Basically copied from: https://gist.github.com/akesling/5358964
"""
def LoadMNISTData(dataset='train', path='data/'):
    # http://yann.lecun.com/exdb/mnist/
    if dataset is 'train':
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is 'test':
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        sys.exit('Error: invalid dataset')


    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        y = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        X = np.fromfile(fimg, dtype=np.uint8)
        X = X.reshape(len(y), rows, cols)


    X = X.astype('float32')
    X /= 255.0

    return X, y

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

'''
Basically copied from:
https://luckydanny.blogspot.com/2016/07/load-cifar-10-dataset-in-python3.html
'''
def LoadCifar10(data_dir):
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic[b'data']
        else:
            train_data = np.vstack((train_data, data_dic[b'data']))
        train_labels += data_dic[b'labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic[b'data']
    test_labels = test_data_dic[b'labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


