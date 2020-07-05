import matplotlib.pyplot as plt
import random as rd
import numpy as np
import pandas as pd

def datacreation(N):
    # Data generation
    data = pd.DataFrame( (np.random.rand(N, 2)*2) - 1, columns=['ix1','ix2'])
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
    data.loc[(iw[0]*data['ix0'] + iw[1]*data['ix1'] + iw[2]*data['ix2']) >= 0, 'iy'] = 1

    return data, iw

def pla(data):
    #Converting dataframe to matrix
    features = np.asarray(data.loc[:, 'ix0':'ix2'])

    # set weights to zero
    w = np.zeros(shape=(1, features.shape[1]))

    # Iterating till convergence
    num = 0
    data['g'] = 5
    missclassified = data[(data['g'] != data['iy'])]
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

def plotcurve(data, wieght):
    #Converting wieght into right coefficient
    coefficients = np.zeros(shape=2)
    coefficients[0] = -wieght[0,1]/wieght[0,2]
    coefficients[1] = -wieght[0,0]/wieght[0,2]

    #Plotting equation and data
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(-1,1,100)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis)
    plt.scatter(np.array(data[data['iy']==1]['ix1']), np.array(data[data['iy']==1]['ix2']), marker='o', label='1')
    plt.scatter(np.array(data[data['iy']==-1]['ix1']), np.array(data[data['iy']==-1]['ix2']), marker='x', label='-1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()