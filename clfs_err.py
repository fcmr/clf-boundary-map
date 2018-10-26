import numpy as np
import joblib

from boundarymap import CLF
from pathlib import Path

#TODO: repeated code, only BASE_DIR and suc changes...

BASE_DIR = 'data/fashionmnist/'
CLFS = ['KNeighborsClassifier', 'Sequential', 'LogisticRegression', 'RandomForestClassifier']

pred = BASE_DIR + 'clfs/fashionmnist_model_'
suc = '_True.pkl'
clfs_f = [pred + c + suc for c in CLFS]


X_train = np.load(BASE_DIR + 'X_sample_bin.npy')
y_train = np.load(BASE_DIR + 'y_sample_bin.npy')
# TODO: y_train contains the values 0 and 9, replace all 9s for 1s?
y_train[y_train == 9] = 1

# TODO: load y_pred if it exists
y_preds_f = [BASE_DIR + 'y_sample_bin_pred_' + c + '.npy' for c in CLFS]
y_pred_exists = [Path(y).is_file() for y in y_preds_f]
print("y_pred_exists")

clfs = [None]*len(CLFS)

y_preds = [None]*len(CLFS)
for i in range(4):
    if y_pred_exists[i] is False:
        if i == 1:
            from keras import load_model
            clfs[i] = CLF(clf=load_model(clfs_f[i]), clf_type="keras_cnn", shape=(28, 28, 1))
        else:
            clfs[i] = CLF(clf=joblib.load(open(clfs_f[i], 'rb')), clf_type="sklearn")
        y_preds[i] = clfs[i].Predict(X_train)
        np.save(y_preds_f[i], y_preds[i])
    else:
        y_preds[i] = np.load(y_preds_f[i])

print("2 class fashion mnist - {} samples".format(len(y_train)))
for i in range(len(CLFS)):
    clf_name = CLFS[i]
    print('\t' + clf_name + " num errors: ", np.sum(y_train != y_preds[i]))

BASE_DIR = 'data/fashionmnist_full/'
X_train = np.load(BASE_DIR + 'X_sample.npy')
y_train = np.load(BASE_DIR + 'y_sample.npy')

pred = BASE_DIR + 'clfs/fashionmnist_model_'
suc = '_False.pkl'
clfs_f = [pred + c + suc for c in CLFS]

y_preds_f = [BASE_DIR + 'y_sample_pred_' + c + '.npy' for c in CLFS]
y_pred_exists = [Path(y).is_file() for y in y_preds_f]

clfs = [None]*len(CLFS)
y_preds = [None]*len(CLFS)
for i in range(4):
    if y_pred_exists[i] is False:
        if i == 1:
            from keras import load_model
            clfs[i] = CLF(clf=load_model(clfs_f[i]), clf_type="keras_cnn", shape=(28, 28, 1))
        else:
            clfs[i] = CLF(clf=joblib.load(open(clfs_f[i], 'rb')), clf_type="sklearn")
        y_preds[i] = clfs[i].Predict(X_train)
        np.save(y_preds_f[i], y_preds[i])
    else:
        y_preds[i] = np.load(y_preds_f[i])

print("10 class fashion mnist - {} samples".format(len(y_train)))
for i in range(len(CLFS)):
    clf_name = CLFS[i]
    print('\t' + clf_name + " num errors: ", np.sum(y_train != y_preds[i]))
