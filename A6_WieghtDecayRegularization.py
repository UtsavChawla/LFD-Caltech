import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression


# Fetches Raw Data from Caltech Website
def fetchdata():
    with open("in.dta", "x") as f_in:
        request_in = requests.get("http://work.caltech.edu/data/in.dta")
        f_in.write(request_in.text)
    with open("out.dta", "x") as f_out:
        request_out = requests.get("http://work.caltech.edu/data/out.dta")
        f_out.write(request_out.text)


# Transforms Data
def transform(data):
    Xin = pd.DataFrame(data=data[:, 0:2], columns=('x1', 'x2'))
    Xin['A'] = 1
    Xin['D'] = Xin['x1'] ** 2
    Xin['E'] = Xin['x2'] ** 2
    Xin['F'] = Xin['x1'] * Xin['x2']
    Xin['G'] = abs(Xin['x1'] - Xin['x2'])
    Xin['H'] = abs(Xin['x1'] + Xin['x2'])
    Xin = Xin[['A', 'x1', 'x2', 'D', 'E', 'F', 'G']]
    Xin.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    Yin = data[:, 2]
    return Xin, Yin


# Converts output of Linear Regression into binary form
def f(x):
    if x >= 0:
        return 1
    else:
        return -1


# Regression with wieght decay
def regress(Xin, Yin, k):
    Lamda = 10 ** k
    Z = Xin.to_numpy()
    Zt = Xin.T.to_numpy()
    return np.matmul(np.linalg.inv(np.add(np.matmul(Zt, Z), Lamda * np.identity(7))), np.matmul(Zt, Yin))


## Preparing Data
# fetchdata()
data_train = np.loadtxt("in.dta")
data_test = np.loadtxt("out.dta")
data_in = transform(data_train)
Xin, Yin = data_in

data_out = transform(data_test)
Xout, Yout = data_out


for k in range(-5,5,1):
    # Regressing with no intercept as A = 1
    W = regress(Xin, Yin, k)

    # Checking error
    Yin_pred = np.dot(Xin, W)
    Yin_pred = np.array([f(xi) for xi in Yin_pred])
    Ein_tb = abs(Yin - Yin_pred) / 2
    Ein = sum(Ein_tb) / len(Ein_tb)

    # Eout
    Yout_pred = np.dot(Xout, W)
    Yout_pred = np.array([f(xi) for xi in Yout_pred])
    Eout_tb = abs(Yout - Yout_pred) / 2
    Eout = sum(Eout_tb) / len(Eout_tb)

    print(k, Ein, Eout)
