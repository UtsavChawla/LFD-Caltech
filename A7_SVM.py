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

    flag = True
    while(flag):
        print("Yay")
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
        if( len(data[data['iy']==1])  not in [0, len(data)] ):
            flag = False

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


N = 3
input = datacreation(N)
data_in = input[0]
iw = input[1]
plotcurve(data_in, iw)