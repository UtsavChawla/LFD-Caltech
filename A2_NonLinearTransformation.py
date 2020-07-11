import matplotlib.pyplot as plt
import random as rd
import numpy as np
import pandas as pd


def datacreation_noise(N, x):
    # Data generation
    data = pd.DataFrame((np.random.rand(N, 2) * 2) - 1, columns=['ix1', 'ix2'])
    data['ix0'] = 1
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    # Assigning y values
    data['iy'] = -1
    data.loc[((data['ix1'] * data['ix1']) + (data['ix2'] * data['ix2']) - 0.6) >= 0, 'iy'] = 1

    # Adding noise
    for i in range(int(0.1 * len(data))):
        data.at[i, 'iy'] = data.iloc[i]['iy'] * -1

    return data


def plotcurve(data, ow):
    # Plotting equation
    x_axis = np.linspace(-1, 1, 100)
    y_axis = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_axis, y_axis)
    F = ow[0] + ow[1] * X + ow[2] * Y + ow[3] * X * Y + ow[4] * (X ** 2) + ow[5] * (Y ** 2)
    plt.contour(X, Y, F, [0])

    # plotting data

    plt.scatter(np.array(data[data['iy'] == 1]['ix1']), np.array(data[data['iy'] == 1]['ix2']), marker='o', label='1')
    plt.scatter(np.array(data[data['iy'] == -1]['ix1']), np.array(data[data['iy'] == -1]['ix2']), marker='x',
                label='-1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def regress(data):
    x = np.asarray(data.loc[:, 'ix0':'ix2'])
    y = np.asarray(data.loc[:, 'iy':'iy'])
    xt = np.dot(np.linalg.inv(np.dot(x.transpose(), x)), x.transpose())
    return np.dot(xt, y).flatten()


def regress_transform(data):
    new = data.copy()
    new.columns = ['a', 'b', 'c', 'y']
    new['d'] = new['b'] * new['c']
    new['e'] = new['b'] * new['b']
    new['f'] = new['c'] * new['c']
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'y']
    newdata = new[cols]

    x = np.asarray(newdata.loc[:, 'a':'f'])
    y = np.asarray(newdata.loc[:, 'y':'y'])
    xt = np.dot(np.linalg.inv(np.dot(x.transpose(), x)), x.transpose())
    return np.dot(xt, y).flatten()


def errorcompute(data, ow):
    features = np.asarray(data.loc[:, 'ix0':'ix2'])
    data['graw'] = np.dot(features, ow.transpose())
    data['g'] = -1
    data.loc[data['graw'] >= 0, 'g'] = 1
    return len(data[(data['g'] != data['iy'])]) / len(data)


def erroroutcompute():
    # Data generation
    data = pd.DataFrame((np.random.rand(1000, 2) * 2) - 1, columns=['ix1', 'ix2'])
    data['ix0'] = 1
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    # Assigning y values using f
    data['iy'] = -1
    data.loc[((data['ix1'] * data['ix1']) + (data['ix2'] * data['ix2']) - 0.6) >= 0, 'iy'] = 1

    # Adding noise
    for i in range(int(0.1 * len(data))):
        data.at[i, 'iy'] = data.iloc[i]['iy'] * -1

    # Assigning y values using g
    data['oy'] = -1
    data.loc[(-1 - (0.05 * data['ix1']) + (0.08 * data['ix2']) + (0.13 * data['ix1'] * data['ix2']) + (
                1.5 * data['ix1'] * data['ix1']) + (1.5 * data['ix2'] * data['ix2'])) >= 0, 'oy'] = 1

    # Returning error
    return len(data[(data['iy'] * data['oy'] < 0)]) / len(data)


####### Direct regression
N = 1000
x = 0.1
runs = 1000
errorsum = 0
for i in range(runs):
    print(i)
    data = datacreation_noise(N, x)
    ow = regress(data)
    errorsum += errorcompute(data, ow)
error_in = errorsum / runs

####### Feature based regression
N = 1000
x = 0.1
runs = 1000
ow_record = np.zeros((runs, 6))
for i in range(runs):
    print(i)
    data = datacreation_noise(N, x)
    ow = regress_transform(data)
    ow_record[i] = ow

s = np.mean(ow_record, axis=0)
# plotcurve(data, ow)

errorsum = 0
for i in range(1000):
    print(i)
    errorsum = errorsum + erroroutcompute()

mape = errorsum/1000