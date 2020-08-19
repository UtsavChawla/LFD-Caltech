import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression


def fetchdata():
    with open("in.dta", "x") as f_in:
        request_in = requests.get("http://work.caltech.edu/data/in.dta")
        f_in.write(request_in.text)
    with open("out.dta", "x") as f_out:
        request_out = requests.get("http://work.caltech.edu/data/out.dta")
        f_out.write(request_out.text)


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


## Preparing Data
# fetchdata()
data_train = np.loadtxt("in.dta")
data_test = np.loadtxt("out.dta")
data_in = transform(data_train)
Xin, Yin = data_in

data_out = transform(data_test)
Xout, Yout = data_out
# No intercept
# reg = LinearRegression().fit(Xin, Yin)
