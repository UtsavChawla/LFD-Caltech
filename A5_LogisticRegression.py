import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import math as mt


# Defining function f(x) = if(iw0*ix0 + iw1*ix1 + iw2*ix2>0) then 1 else 0 ; returns weight
def inputfunc():
    xf = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
    yf = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
    coefficients = np.polyfit(xf, yf, 1)
    iw = np.zeros(shape=3)
    iw[2] = 1
    iw[1] = -coefficients[0]
    iw[0] = -coefficients[1]

    return iw


# generating labeled data from function
def labeleddata(N, w):
    data = pd.DataFrame((np.random.rand(N, 2) * 2) - 1, columns=['ix1', 'ix2'])
    data['ix0'] = 1
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data['iy'] = -1
    data.loc[(w[0] * data['ix0'] + w[1] * data['ix1'] + w[2] * data['ix2']) > 0, 'iy'] = 1
    return data


# Plotting curve
def plotlinearcurve(data, wieght):
    coefficients = np.zeros(shape=2)
    coefficients[0] = -wieght[1] / wieght[2]
    coefficients[1] = -wieght[0] / wieght[2]
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(-1, 1, 100)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis)
    plt.scatter(np.array(data[data['iy'] == 1]['ix1']), np.array(data[data['iy'] == 1]['ix2']), marker='o', label='1')
    plt.scatter(np.array(data[data['iy'] == 0]['ix1']), np.array(data[data['iy'] == 0]['ix2']), marker='x', label='0')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


# Logistic regression to predict outcome
def logisticregression(data):
    ow = np.zeros(3)
    eta = 0.01

    delta = 1
    runs = 0

    while (delta > 0.01):

        if(runs % 100 == 0):
            print(runs)

        ow_old = ow
        arr = np.array(range(100))
        np.random.shuffle(arr)
        for p in arr:
            xn = np.array(data.iloc[p][0:3])
            yn = np.array(data.iloc[p][3:4])
            DerEn = (-1) * (yn * xn) / (1 + np.exp(yn * np.dot(ow.T, xn)))
            ow = ow - (eta * DerEn)

        delta = np.linalg.norm(ow - ow_old)
        runs = runs + 1

    return ow, runs


# Execution
execs = 10
N = 100
runs = 0

for i in range(execs):
    print(i)
    iw = inputfunc()
    data = labeleddata(N, iw)
    output = logisticregression(data)
    ow = output[0]
    runs = runs + output[1]

runs = runs/execs



# calculating cross entropy error
#eN = 1000
#edata = labeleddata(1000, iw)
#edata['Error'] = 0.1

#for p in range(len(edata)):
 #   xn = np.array(edata.iloc[p][0:3])
  #  yn = np.array(edata.iloc[p][3:4])
   # E = np.log(1 + np.exp(-yn * np.dot(ow.T, xn))).mean()
    #edata.at[p,'Error'] = E

#error = edata['Error'].mean()