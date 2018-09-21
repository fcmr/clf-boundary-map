import argparse
import os
import random as rn

from glob import glob
import joblib
import keras
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def load_dataset(dataset_name, dataset_type='full', is_binary=True, img=False):
    data_dir = os.path.join('data', dataset_name)

    if dataset_type not in ['full', 'sample']:
        raise "dataset_type must be 'full' or 'sample'"

    if is_binary:
        file_ext = '_%s_bin.npy' % dataset_type
    else:
        file_ext = '_%s.npy' % dataset_type

    if img:
        X = np.load(os.path.join(data_dir, 'X_img%s' % file_ext))
        X = np.expand_dims(X, axis=4)
    else:
        X = np.load(os.path.join(data_dir, 'X%s' % file_ext))

    y = np.load(os.path.join(data_dir, 'y%s' % file_ext))

    return X, y

def predict(clf, X):
    y_pred = clf.predict(X)

    if len(y_pred.shape) > 1:
        return np.argmax(y_pred, axis=1)
    else:
        return y_pred


def get_cnn():
    def get_model(input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_uniform', bias_initializer=Constant(0.01), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_uniform', bias_initializer=Constant(0.01), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, kernel_initializer='he_uniform', bias_initializer=Constant(0.01), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, kernel_initializer='he_uniform', bias_initializer=Constant(0.01), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        return model

    return get_model


def train_model(model, dataset_name, is_binary=True):
    if hasattr(model, '__name__') and model.__name__ == 'get_model':
        X, y = load_dataset(dataset_name, 'full', is_binary, img=True)
        y = np_utils.to_categorical(y)
        input_shape = X.shape[1:]
        num_classes = y.shape[1]
        cnn = model(input_shape, num_classes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=False, mode='max')
        cnn.fit(X_train, y_train, verbose=False, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[stopper])
        return cnn, cnn.evaluate(X_test, y_test, verbose=False)[1]
    else:
        X, y = load_dataset(dataset_name, 'full', is_binary, img=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        model.fit(X_train, y_train)
        return model, model.score(X_test, y_test)

def score_model(model, dataset_name, is_binary=True):
    if model.__class__.__name__ == 'Sequential':
        X, y = load_dataset(dataset_name, 'full', is_binary, img=True)
        y = np_utils.to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        return model.evaluate(X_test, y_test, verbose=False)[1]
    else:
        X, y = load_dataset(dataset_name, 'full', is_binary, img=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        return model.score(X_test, y_test)


def save_model(model, model_name, dataset_name, output_dir, is_binary=True):
    model_file_name = os.path.join(output_dir, '%s_model_%s_%s' % (dataset_name, model_name, str(is_binary)))

    if model.__class__.__name__ == 'Sequential':
        model_file_name = '%s%s' % (model_file_name, '.h5')
        model.save(model_file_name)
    else:
        model_file_name = '%s%s' % (model_file_name, '.pkl')
        joblib.dump(model, model_file_name)


def load_model(model_file_name):
    if model_file_name[-3:] == '.h5':
        session_conf = tf.ConfigProto()
        session_conf.intra_op_parallelism_threads = 1
        session_conf.inter_op_parallelism_threads = 1
        session_conf.gpu_options.allow_growth = True

        K.clear_session()
        
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        model = keras.models.load_model(model_file_name)
    else:
        model = joblib.load(model_file_name)

    return model


def run_classifiers(dataset_name, output_dir, is_binary):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # begin set seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)

    tf.set_random_seed(42)
    # end set seed

    models = [  LogisticRegression(),
                KNeighborsClassifier(n_neighbors=5),
                RandomForestClassifier(n_estimators=50),
                get_cnn()]

    for model in models:
        trained_model, acc = train_model(model, dataset_name, is_binary)
        print('%s - %s: %.4f' % (dataset_name, trained_model.__class__.__name__, acc))
        save_model(trained_model, trained_model.__class__.__name__, dataset_name, output_dir, is_binary)

def score_classifiers(dataset_name, classifier_dir, is_binary):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # begin set seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)

    tf.set_random_seed(42)
    # end set seed

    model_files = glob(classifier_dir + '/%s_model_*_%s.*' % (dataset_name, str(is_binary)))

    for model_file in model_files:
        model = load_model(model_file)
        acc = score_model(model, dataset_name, is_binary)
        print('%s - %s: %.4f' % (dataset_name, model_file, acc))
