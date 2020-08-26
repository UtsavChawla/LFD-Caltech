import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


#Returning data with Linear Transformation
def transform(data):
    Xin = pd.DataFrame(data=data[:, 0:2], columns=('x1', 'x2'))
    Xin['A'] = 1
    Xin['D'] = Xin['x1'] ** 2
    Xin['E'] = Xin['x2'] ** 2
    Xin['F'] = Xin['x1'] * Xin['x2']
    Xin['G'] = abs(Xin['x1'] - Xin['x2'])
    Xin['H'] = abs(Xin['x1'] + Xin['x2'])
    Xin = Xin[['A', 'x1', 'x2', 'D', 'E', 'F', 'G', 'H']]
    Xin.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    Yin = data[:, 2]
    return Xin, Yin

def f(x):
    if x>0:
        return 1
    else:
        return -1

def error(Xval_n, Yval, Xtest_n, Ytest):
    # Eval
    Yval_pred = np.dot(Xval_n, reg.coef_)
    Yval_pred = np.array([f(xi) for xi in Yval_pred])
    Eval_tb = abs(Yval - Yval_pred) / 2
    Eval = sum(Eval_tb) / len(Eval_tb)

    # Etest
    Ytest_pred = np.dot(Xtest_n, reg.coef_)
    Ytest_pred = np.array([f(xi) for xi in Ytest_pred])
    Etest_tb = abs(Ytest - Ytest_pred) / 2
    Etest = sum(Etest_tb) / len(Etest_tb)

    return Eval, Etest


#Getting raw data
data_full = np.loadtxt("in.dta")
n = 25
data_train = data_full[0:n]
data_val = data_full[n:]
data_test = np.loadtxt("out.dta")

Xtrain, Ytrain = transform(data_train)
Xval, Yval = transform(data_val)
Xtest, Ytest = transform(data_test)

#Learning from Data
for k in range(3,8,1):
    Xval_n = Xval.iloc[:,0:k+1]
    Xtrain_n = Xtrain.iloc[:,0:k+1]
    Xtest_n = Xtest.iloc[:,0:k+1]

    reg = LinearRegression(fit_intercept=False).fit(Xtrain_n, Ytrain)
    print(k,error(Xval_n, Yval, Xtest_n, Ytest))