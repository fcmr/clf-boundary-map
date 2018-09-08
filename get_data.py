import os
import shutil
import tempfile
import zipfile
from glob import glob
import numpy as np
import wget
from skimage import io, transform
from keras import datasets as kdatasets
from keras import applications
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def save_dataset(base_dir, name, loader, bin_classes, sample_size=0.2):
    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    (X, y), (_, _) = loader()
    y = y.squeeze()

    X = X.astype('float32') / 255.0

    if len(X.shape) > 3:
        X = X[:,:,:,1] #takes only Green channel from color images

    X_flat = X.reshape((-1, X.shape[1]**2))

    X_bin = X[np.isin(y, bin_classes)]
    X_flat_bin = X_flat[np.isin(y, bin_classes)]
    y_bin = y[np.isin(y, bin_classes)].copy()

    bin_false = np.min(y_bin)
    bin_true  = np.max(y_bin)

    y_bin[y_bin == bin_false] = 0
    y_bin[y_bin == bin_true] = 1

    print(name, 'full dataset', X_flat.shape, X.shape, y.shape)
    np.save(os.path.join(dir_name, 'X_full.npy'), X_flat)
    np.save(os.path.join(dir_name, 'X_img_full.npy'), X)
    np.save(os.path.join(dir_name, 'y_full.npy'), y.squeeze())

    print(name, 'full dataset (binary)', X_flat_bin.shape, X_bin.shape, y_bin.shape)
    np.save(os.path.join(dir_name, 'X_full_bin.npy'), X_flat_bin)
    np.save(os.path.join(dir_name, 'X_img_full_bin.npy'), X_bin)
    np.save(os.path.join(dir_name, 'y_full_bin.npy'), y_bin.squeeze())

    _, X_sample, _, y_sample = train_test_split(X, y, test_size=sample_size, stratify=y, random_state=42)
    X_sample_flat = X_sample.reshape((-1, X_sample.shape[1]**2))

    X_sample_bin = X_sample[np.isin(y_sample, bin_classes)]
    X_sample_flat_bin = X_sample_flat[np.isin(y_sample, bin_classes)]
    y_sample_bin = y_sample[np.isin(y_sample, bin_classes)]

    print(name, 'sampled dataset', X_sample_flat.shape, X_sample.shape, y_sample.shape)
    np.save(os.path.join(dir_name, 'X_sample.npy'), X_sample_flat)
    np.save(os.path.join(dir_name, 'X_img_sample.npy'), X_sample)
    np.save(os.path.join(dir_name, 'y_sample.npy'), y_sample.squeeze())

    print(name, 'sampled dataset (binary)', X_sample_flat_bin.shape, X_sample_bin.shape, y_sample_bin.shape)
    np.save(os.path.join(dir_name, 'X_sample_bin.npy'), X_sample_flat_bin)
    np.save(os.path.join(dir_name, 'X_img_sample_bin.npy'), X_sample_bin)
    np.save(os.path.join(dir_name, 'y_sample_bin.npy'), y_sample_bin.squeeze())


# def process_cifar10(base_dir):
#     save_dataset(base_dir, 'cifar10', kdatasets.cifar10.load_data, [4, 9])


def process_mnist(base_dir):
    save_dataset(base_dir, 'mnist', kdatasets.mnist.load_data, [0, 1])


def process_fashionmnist(base_dir):
    save_dataset(base_dir, 'fashionmnist', kdatasets.fashion_mnist.load_data, [0, 9])


if __name__ == '__main__':
    base_dir = './data'

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        globals()[func](base_dir)
