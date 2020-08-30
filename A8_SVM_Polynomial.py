import numpy as np
import pandas as pd
import requests
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score


def fetchdata():
    with open("train.dta", "x") as f_in:
        request_in = requests.get("http://www.amlbook.com/data/zip/features.train")
        f_in.write(request_in.text)
    with open("test.dta", "x") as f_out:
        request_out = requests.get("http://www.amlbook.com/data/zip/features.test")
        f_out.write(request_out.text)


def labeldata_1vsall(Y, k):
    Yn = Y.copy()
    Yn[Yn != k] = -1
    Yn[Yn == k] = 1
    return Yn


def errorcompute(model, Xtrain, Ytrain):
    Ytrain_pred = model.predict(Xtrain)
    temp = Ytrain - Ytrain_pred
    return 1 - (len(temp[temp == 0]) / len(temp))


##Execution
# Preparing Data and labeling for 1 vs 5 classifier
# fetchdata()
data_train = np.loadtxt("train.dta")
data_test = np.loadtxt("test.dta")
tp1 = data_train[data_train[:,0]==1]
tp2 = data_train[data_train[:,0]==5]
tp2[:,0] = -1
data_train = np.append(tp1, tp2, axis=0)

tp1 = data_test[data_test[:,0]==1]
tp2 = data_test[data_test[:,0]==5]
tp2[:,0] = -1
data_test = np.append(tp1, tp2, axis=0)
del tp1, tp2

Xtrain = data_train[:, 1:3]
Ytrain = data_train[:, 0]
Xtest = data_test[:, 1:3]
Ytest = data_test[:, 0]
del data_train, data_test

# Modeling
mat = pd.DataFrame({'C': [0.0001,0.001,0.01,0.1,1], 'Freq': [0, 0, 0, 0, 0], 'Ecv': [0.0, 0.0, 0.0, 0.0, 0.0]})
mat = mat.set_index('C')

runs = 100
for i in range(runs):
    k_fold = KFold(n_splits=10, shuffle=True)
    print(i)
    for C in [0.0001,0.001,0.01,0.1,1.0]:
        model = svm.SVC(C=C, degree=2, kernel='poly', coef0=1, gamma=1)
        scores = cross_val_score( model, Xtrain, Ytrain, cv = k_fold)
        Ecv = 1 - scores.mean()
        mat.at[C, 'Ecv'] = Ecv
    mat.at[mat[['Ecv']].idxmin()[0], 'Freq']= mat.at[mat[['Ecv']].idxmin()[0], 'Freq'] + 1