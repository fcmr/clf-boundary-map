# prepare_data.py: Creates a single file containing the shuffled dataset,
# a subset of projected points and a trained classifier.

import numpy as np
np.random.seed(1)

import data
import lamp

from sklearn import manifold
from sklearn import linear_model, svm, neighbors, neural_network

import pickle
import json

from time import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def TrainTestSplit(X_orig, y_orig, split_sz):
    new_idx = np.random.permutation(X_orig.shape[0])
    X, y = X_orig[new_idx], y_orig[new_idx]

    train_size = int(X.shape[0]*split_sz)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test

# clfs: array of 
def SaveData(name, path, train_X, train_y, test_X, test_y, proj, clfs):
    train_X_path = path + name + "_X_train.npy"
    train_y_path = path + name + "_y_train.npy"
    test_X_path =  path + name + "_X_test.npy"
    test_y_path =  path + name + "_y_test.npy"
    proj_path =    path + name + "_proj.npy"

    np.save(train_X_path, train_X)
    np.save(train_y_path, train_y)
    np.save(test_X_path,  test_X)
    np.save(test_y_path,  test_y)
    np.save(proj_path,    proj)

    data = {'X_train' : train_X_path,
            'y_train' : train_y_path,
            'X_test'  : test_X_path,
            'y_test'  : test_y_path,
            'proj'    : proj_path,
            'clfs'    : clfs}
     
    with open(path + name + ".json", 'w') as outfile:
        json.dump(data, outfile)

def Toy():
    # Load TOY dataset
    start = time()
    print("Reading TOY dataset...")
    toy_X, toy_y = data.LoadZeroOneData(); 
    print("\tFinished reading TOY dataset...", time() - start)
    
    start = time()
    print("TrainTestSplit for TOY...")
    toy_X_train, toy_y_train, toy_X_test, toy_y_test = TrainTestSplit(toy_X, toy_y, 0.7)
    print("\tFinished TrainTestSplit...", time() - start)

    # Uses LAMP to project the entire dataset
    start = time()
    print("LAMP projecting TOY dataset...")
    toy_proj = lamp.lamp2d(toy_X_train)
    print("\tFinished projecting...", time() - start)

    start = time()
    print("Training classifier...")
    toy_lr = linear_model.LogisticRegression()
    toy_lr.fit(toy_X_train, toy_y_train)
    print("\tAccuracy on test data: ", toy_lr.score(toy_X_test, toy_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Saving data for TOY...")
    clfs = ["data/toy/toy_logistic_regression.pkl"]
    with open('data/toy/toy_logistic_regression.pkl', 'wb') as f:
        pickle.dump(toy_lr, f)
    SaveData("toy", "data/toy/", toy_X_train, toy_y_train, toy_X_test,
             toy_y_test, toy_proj, clfs)
    
    print("\tFinished saving data...", time() - start)

def Wine():
    start = time()
    print("Reading WINE dataset...")
    wine_X, wine_y = data.LoadWineData()
    print("\tFinished reading WINE dataset...", time() - start)

    start = time()
    print("TrainTestSplit for WINE...")
    wine_X_train, wine_y_train, wine_X_test, wine_y_test = TrainTestSplit(wine_X, wine_y, 0.7)
    print("\tFinished TrainTestSplit...", time() - start)

    # Uses LAMP to project the entire dataset
    start = time()
    print("LAMP projecting WINE dataset...")
    wine_proj = lamp.lamp2d(wine_X_train)
    print("\tFinished projecting...", time() - start)

    start = time()
    print("Training classifier...")
    wine_lr = linear_model.LogisticRegression()
    wine_lr.fit(wine_X_train, wine_y_train)
    print("\tAccuracy on test data: ", wine_lr.score(wine_X_test, wine_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Saving data for WINE...")
    clfs = ["data/wine/wine_logistic_regression.pkl"]
    with open('data/wine/wine_logistic_regression.pkl', 'wb') as f:
        pickle.dump(wine_lr, f)

    SaveData("wine", "data/wine/", wine_X_train, wine_y_train, wine_X_test,
             wine_y_test, wine_proj, clfs)
    print("\tFinished saving data...", time() - start)

def Segmentation():
    start = time()
    print("Reading SEGMENTATION dataset...")
    seg_X, seg_y = data.LoadSegmentationData()
    print("\tFinished reading SEGMENTATION dataset...", time() - start)


    start = time()
    print("TrainTestSplit for SEGMENTATION...")
    seg_X_train, seg_y_train, seg_X_test, seg_y_test = TrainTestSplit(seg_X, seg_y, 0.7)
    print("\tFinished TrainTestSplit...", time() - start)

    # Uses LAMP to project the entire dataset
    start = time()
    print("LAMP projecting SEGMENTATION dataset...")
    seg_proj = lamp.lamp2d(seg_X_train, 150, 8.0)
    print("\tFinished projecting...", time() - start)

    start = time()
    print("Training classifier LogisticRegression...")
    seg_lr = linear_model.LogisticRegression()
    seg_lr.fit(seg_X_train, seg_y_train)
    print("\tAccuracy on test data: ", seg_lr.score(seg_X_test, seg_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Training classifier SVM...")
    seg_svm = svm.SVC()
    seg_svm.fit(seg_X_train, seg_y_train)
    print("\tAccuracy on test data: ", seg_svm.score(seg_X_test, seg_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Training classifier KNN...")
    seg_knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
    seg_knn5.fit(seg_X_train, seg_y_train)
    print("\tAccuracy on test data: ", seg_knn5.score(seg_X_test, seg_y_test))
    print("\tFinished training classifier...", time() - start)


    start = time()
    print("Saving data for SEGMENTATION...")
    clfs = ["data/segmentation/seg_logistic_regression.pkl",
            "data/segmentation/seg_svm.pkl",
            "data/segmentation/seg_knn5.pkl"]

    with open('data/segmentation/seg_logistic_regression.pkl', 'wb') as f:
        pickle.dump(seg_lr, f)
    with open('data/segmentation/seg_svm.pkl', 'wb') as f:
        pickle.dump(seg_svm, f)
    with open('data/segmentation/seg_knn5.pkl', 'wb') as f:
        pickle.dump(seg_knn5, f)

    SaveData("seg", "data/segmentation/", seg_X_train, seg_y_train, seg_X_test,
             seg_y_test, seg_proj, clfs)
    print("\tFinished saving data...", time() - start)

def CNNModel(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=0.1),
                  metrics=['accuracy'])
    return model

def CNNModel2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])
    return model



def MNIST():
    start = time()
    print("Reading MNIST dataset...")
    X_train, y_train = data.LoadMNISTData('train', 'data/mnist/orig/')
    new_idx = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[new_idx], y_train[new_idx]
    X_train_base = np.copy(X_train)
    y_train_base = np.copy(y_train)

    X_test, y_test = data.LoadMNISTData('test', 'data/mnist/orig')
    X_test_base = np.copy(X_test)
    y_test_base = np.copy(y_test)
    print("\tFinished reading dataset...", time() - start)

    projection_size = 2000
    X_proj = np.copy(X_train[:projection_size])
    X_proj = np.reshape(X_proj, (X_proj.shape[0], X_proj.shape[1]*X_proj.shape[2]))

    # Uses LAMP to project projection_size points from the dataset
    start = time()
    print("LAMP projecting MNIST dataset...")
    proj_lamp = lamp.lamp2d(X_proj, 150, 10.0)
    print("\tFinished projecting...", time() - start)

    # Uses t-SNE to project projection_size points from the dataset
    start = time()
    print("t-SNE projecting MNIST dataset...")
    tsne = manifold.TSNE(n_components=2, perplexity=20.0)
    proj_tsne = tsne.fit_transform(X_proj)
    proj_tsne = (proj_tsne - proj_tsne.min(axis=0))/(proj_tsne.max(axis=0) - proj_tsne.min(axis=0))
    print("\tProjection finished: ", time() - start)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train.reshape((X_train.shape[0],) + input_shape)
    X_test = X_test.reshape((X_test.shape[0],) + input_shape )
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    start = time()
    print("Training classifier CNN...")
    clf1 = CNNModel(input_shape, 10)
    clf1.fit(X_train, y_train, batch_size=128, epochs=14, verbose=1,
             validation_data=(X_test, y_test))
    print("\tAccuracy on test data: ", clf1.evaluate(X_test, y_test, verbose=0))
    print("\tFinished training classifier...", time() - start)
    clf1.save_weights("data/mnist/mnist_cnn1.hdf5")

    start = time()
    print("Training classifier CNN 2...")
    clf2 = CNNModel2(input_shape, 10)

    print("\tEpoch 1:")
    clf2.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1,
             validation_data=(X_test, y_test))
    clf2.save_weights("data/mnist/mnist_cnn2_1e.hdf5")
    print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

    print("\tEpoch 5:")
    clf2.fit(X_train, y_train, batch_size=128, epochs=4, verbose=1,
             validation_data=(X_test, y_test))
    clf2.save_weights("data/mnist/mnist_cnn2_5e.hdf5")
    print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

    print("\tEpoch 10:")
    clf2.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1,
             validation_data=(X_test, y_test))
    clf2.save_weights("data/mnist/mnist_cnn2_10e.hdf5")
    print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

    print("\tEpoch 50:")
    clf2.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1,
             validation_data=(X_test, y_test))
    clf2.save_weights("data/mnist/mnist_cnn2_50e.hdf5")
    print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))
    print("\tFinished training classifier...", time() - start)

    print("Saving data for MNIST...")
    weights = ["data/mnist/mnist_cnn1.hdf5",
               "data/mnist/mnist_cnn2_1e.hdf5",
               "data/mnist/mnist_cnn2_5e.hdf5",
               "data/mnist/mnist_cnn2_10e.hdf5",
               "data/mnist/mnist_cnn2_50e.hdf5"]

    # Save the model architecture
    clfs = ["data/mnist/cnn1_architecture.json",
            "data/mnist/cnn2_architecture.json"]    

    with open(clfs[0], 'w') as f:
        f.write(clf1.to_json())

    with open(clfs[1], 'w') as f:
        f.write(clf2.to_json())

    path = "data/mnist/"
    name = "mnist"
    train_X_path = path + name + "_X_train.npy"
    train_y_path = path + name + "_y_train.npy"
    test_X_path =  path + name + "_X_test.npy"
    test_y_path =  path + name + "_y_test.npy"
    proj_path1 =   path + name + "_lamp_proj.npy"
    proj_path2 =   path + name + "_tsne_proj.npy"

    np.save(train_X_path, X_train_base)
    np.save(train_y_path, y_train_base)
    np.save(test_X_path,  X_test_base)
    np.save(test_y_path,  y_test_base) 
    np.save(proj_path1,   proj_lamp)
    np.save(proj_path2,   proj_tsne)

    data_json = {'X_train' : train_X_path,
                 'y_train' : train_y_path,
                 'X_test'  : test_X_path,
                 'y_test'  : test_y_path,
                 'proj1'   : proj_path1,
                 'proj2'   : proj_path2,
                 'clfs'    : clfs,
                 'weights' : weights }
     
    with open(path + name + ".json", 'w') as outfile:
        json.dump(data_json, outfile)

    print("\tFinished saving data...", time() - start)

def main():
    #Toy()
    #Wine()
    #Segmentation()
    MNIST()

if __name__ == "__main__":
    main()
