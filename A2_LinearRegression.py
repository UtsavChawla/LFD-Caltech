import matplotlib.pyplot as plt
import random as rd
import numpy as np
import pandas as pd


def datacreation(N):
    # Data generation
    data = pd.DataFrame((np.random.rand(N, 2) * 2) - 1, columns=['ix1', 'ix2'])
    data['ix0'] = 1
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    # Defining function f (iw0*ix0 + iw1*ix1 + iw2*ix2 = 0)
    xf = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
    yf = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
    coefficients = np.polyfit(xf, yf, 1)
    iw = np.zeros(shape=3)
    iw[2] = 1
    iw[1] = -coefficients[0]
    iw[0] = -coefficients[1]

    # Assigning y values
    data['iy'] = -1
    data.loc[(iw[0] * data['ix0'] + iw[1] * data['ix1'] + iw[2] * data['ix2']) >= 0, 'iy'] = 1

    return data, iw


def plotcurve(data, wieght):
    # Converting wieght into right coefficient
    coefficients = np.zeros(shape=2)
    coefficients[0] = -wieght[1] / wieght[2]
    coefficients[1] = -wieght[0] / wieght[2]

    # Plotting equation and data
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(-1, 1, 100)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis)
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


def errorcompute(data,ow):
    features = np.asarray(data.loc[:, 'ix0':'ix2'])
    data['graw'] = np.dot(features, ow.transpose())
    data['g'] = -1
    data.loc[data['graw'] >= 0, 'g'] = 1
    return len(data[(data['g'] != data['iy'])]) / len(data)


def erroroutcompute(iw_record,ow_record):
    errorsum = 0

    sample = pd.DataFrame((np.random.rand(1000, 2) * 2) - 1, columns=['ix1', 'ix2'])
    sample['ix0'] = 1
    cols = sample.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    sample = sample[cols]
    features = np.asarray(sample.loc[:, 'ix0':'ix2'])

    for i in range(len(iw_record)):
        print(i)
        iw = iw_record[i]
        ow = ow_record[i]
        sample['fraw'] = np.dot(features, iw.transpose())
        sample['graw'] = np.dot(features, ow.transpose())
        errorsum += len(sample[(sample['fraw'] * sample['graw'] < 0)]) / len(sample)

    return float(errorsum/len(iw_record))


def pla(data, w):
    #Converting dataframe to matrix
    features = np.asarray(data.loc[:, 'ix0':'ix2'])
    # set weights to zero

    # Iterating till convergence
    num =0
    data['graw'] = np.dot(features, w.transpose())
    data['g'] = -1
    data.loc[data['graw'] >= 0, 'g'] = 1
    missclassified = data[(data['g'] != data['iy'])]
    missclassified = missclassified.reset_index(drop=True)

    while(len(missclassified) > 0):
        data['graw'] = np.dot(features, w.transpose())
        data['g'] = -1
        data.loc[data['graw'] >= 0, 'g'] = 1
        missclassified = data[(data['g'] != data['iy'])]
        missclassified = missclassified.reset_index(drop=True)
        if(num>1000):
            print("Fcuk")
            break
        if(len(missclassified) == 0):
            break
        num = num + 1
        index = rd.randint(0, len(missclassified) - 1)
        tp = np.asarray(missclassified.loc[:, 'ix0':'ix2'])
        if(missclassified.iloc[index]['g'] == 1):
            w = w - tp[index]
        else:
            w = w + tp[index]

    return w,num

####### Regression
N = 10
runs = 1000
iw_record = np.zeros((runs,3))
ow_record = np.zeros((runs,3))

errorsum = 0
for i in range(runs):
    print(i)
    input = datacreation(N)
    data = input[0]
    iw = input[1]
    ow = regress(data)
    errorsum += errorcompute(data,ow)
    iw_record[i] = iw
    ow_record[i] = ow

error_in = errorsum/runs
error_out = erroroutcompute(iw_record, ow_record)

####### Regression followed by PLA
N=10
runs = 1000
num = 0

for i in range(runs):
    print(i)
    input = datacreation(N)
    data = input[0]
    iw = input[1]
    ow = regress(data)
    output = pla(data, ow)
    xw = output[0]
    num = num + output[1]

num = num / 1000