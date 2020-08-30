import numpy as np
import requests
from sklearn import svm


def fetchdata():
    with open("train.dta", "x") as f_in:
        request_in = requests.get("http://www.amlbook.com/data/zip/features.train")
        f_in.write(request_in.text)
    with open("test.dta", "x") as f_out:
        request_out = requests.get("http://www.amlbook.com/data/zip/features.test")
        f_out.write(request_out.text)


def labeldata(Y, k):
    Yn = Y.copy()
    Yn[Yn != k] = -1
    Yn[Yn == k] = 1
    return Yn


def errorcompute(model, Xtrain, Ytrain):
    Ytrain_pred = model.predict(Xtrain)
    temp = Ytrain - Ytrain_pred
    return 1 - (len(temp[temp == 0]) / len(temp))


##Execution
# Preparing Data
# fetchdata()
data_train = np.loadtxt("train.dta")
data_test = np.loadtxt("test.dta")
Xtrain = data_train[:, 1:3]
Ytrain_base = data_train[:, 0]
Xtest = data_test[:, 1:3]
Ytest_base = data_test[:, 0]
del data_train, data_test

for k in range(10):
    # Labeling Data
    Ytrain = labeldata(Ytrain_base, k)
    Ytest = labeldata(Ytest_base, k)

    # Modeling
    model = svm.SVC(C=0.01, degree=2, kernel='poly', coef0=1, gamma=1)
    model.fit(Xtrain, Ytrain)
    error = errorcompute(model, Xtrain, Ytrain)
    print(k, error, len(model.support_vectors_))
