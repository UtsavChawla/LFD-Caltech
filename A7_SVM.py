import matplotlib.pyplot as plt
import random as rd
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers


def datacreation(N):
    # Data generation
    data = pd.DataFrame((np.random.rand(N, 2) * 2) - 1, columns=['ix1', 'ix2'])
    data['ix0'] = 1
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    flag = True
    while (flag):
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
        if (len(data[data['iy'] == 1]) not in [0, len(data)]):
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


def pla(data):
    # Converting dataframe to matrix
    features = np.asarray(data.loc[:, 'ix0':'ix2'])
    # set weights to zero
    w = np.zeros(3)

    # Iterating till convergence
    num = 0
    data['graw'] = np.dot(features, w.transpose())
    data['g'] = -1
    data.loc[data['graw'] >= 0, 'g'] = 1
    missclassified = data[(data['g'] != data['iy'])]
    missclassified = missclassified.reset_index(drop=True)

    while (len(missclassified) > 0):
        data['graw'] = np.dot(features, w.transpose())
        data['g'] = -1
        data.loc[data['graw'] >= 0, 'g'] = 1
        missclassified = data[(data['g'] != data['iy'])]
        missclassified = missclassified.reset_index(drop=True)
        if (num > 1000):
            print("Fcuk")
            break
        if (len(missclassified) == 0):
            break
        num = num + 1
        index = rd.randint(0, len(missclassified) - 1)
        tp = np.asarray(missclassified.loc[:, 'ix0':'ix2'])
        if (missclassified.iloc[index]['g'] == 1):
            w = w - tp[index]
        else:
            w = w + tp[index]

    return w, num


def svm(data_in, N):
    ## SVM using QP on dual soln
    # Separating X and Y
    X = np.array(data_in.loc[:, 'ix1':'ix2'])
    Y = np.array(data_in.loc[:, 'iy'])

    # Creating alpha matrix
    mat = []
    for row in range(N):
        for col in range(N):
            val = Y[row] * Y[col] * np.dot(X[row].T, X[col])
            mat.append(val)
    mat = np.array(mat).reshape((N, N))

    # Forming matrices for solving
    P = matrix(mat, tc='d')
    q = matrix(-np.ones(N), tc='d')
    b = matrix(0, tc='d')
    A = matrix(Y, tc='d').trans()
    h = matrix(np.zeros(N), tc='d')
    G = matrix(-np.identity(N), tc='d')

    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(list(sol['x']))

    # Calculating W and separating support vectors
    w = np.zeros(2)
    sv_ids = []
    for i in range(N):
        w += alpha[i] * Y[i] * X[i]
        if (alpha[i] > 0.001):
            sv_ids.append(i)

    # calculating b
    bid = sv_ids[0]
    b = (1 / Y[bid]) - np.dot(w.T, X[bid])

    # final wieghts
    ow_svm = np.insert(w, 0, b)
    num_sv = len(sv_ids)

    return ow_svm, num_sv


############
## Execution
N = 10
input = datacreation(N)
data_in = input[0]
iw = input[1]
del input
# plotcurve(data_in, iw)

## PLA
ow_pla = pla(data_in)[0]
#plotcurve(data_in, ow_pla)

## SVM
out = svm(data_in, N)
ow_svm = out[0]
n_sv = out[1]
#plotcurve(data_in, ow_svm)
