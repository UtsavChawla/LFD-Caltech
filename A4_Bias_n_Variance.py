import numpy as np
import pandas as pd
import math as mt
from sklearn.linear_model import LinearRegression


## Returns data
def sindata(N):
    # Row 0 is x and Row 1 is y
    data = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(N, 2)), columns=list('xy'))
    data['y'] = data.apply(lambda row: mt.sin(3.14 * row['x']), axis=1)
    return data


## Returns output coffecients in a tuple
def learn(data, ftype):
    if (ftype == 'a'):
        ## Linear regression with x -> 0
        x = np.array(data['x'] * 0).reshape((-1, 1))
        y = np.array(data['y'])
        model = LinearRegression().fit(x, y)
    elif (ftype == 'b'):
        ## Linear regression with normal x
        x = np.array(data['x']).reshape((-1, 1))
        y = np.array(data['y'])
        model = LinearRegression(fit_intercept=False).fit(x, y)
    elif (ftype == 'c'):
        ## Linear regression with normal x
        x = np.array(data['x']).reshape((-1, 1))
        y = np.array(data['y'])
        model = LinearRegression().fit(x, y)
    elif (ftype == 'd'):
        ## Linear regression with normal x
        x = np.array(data['x'] ** 2).reshape((-1, 1))
        y = np.array(data['y'])
        model = LinearRegression(fit_intercept=False).fit(x, y)
    elif (ftype == 'e'):
        ## Linear regression with normal x
        x = np.array(data['x'] ** 2).reshape((-1, 1))
        y = np.array(data['y'])
        model = LinearRegression().fit(x, y)

    return model.coef_[0], model.intercept_


## Returns output mean coefficients in array
def gbar(Datasize, ftype):
    runs = 1000
    return np.mean(np.array([learn(sindata(Datasize), ftype) for i in range(runs)]), axis=0)


## Returns float bias
def cal_bias(output, ftype):
    a = output[0]
    b = output[1]
    sett = pd.DataFrame(np.arange(-1, 1, 0.02), columns=list('x'))

    if (ftype in list('abc')):
        sett['bias'] = sett.apply(lambda row: (a * row['x'] + b - mt.sin(3.14 * row['x'])) ** 2, axis=1)
    else:
        sett['bias'] = sett.apply(lambda row: (a * (row['x'] ** 2) + b - mt.sin(3.14 * row['x'])) ** 2, axis=1)

    return sett['bias'].mean()


## Returns float variance
def cal_var(output, Datasize, ftype):
    am = output[0]
    bm = output[1]

    var = 0
    runs = 1000
    for i in range(runs):
        data = sindata(Datasize)
        if (ftype in ('de')):
            data['x'] = data['x'] ** 2
        temp = learn(data, ftype)
        a = temp[0]
        b = temp[1]
        data['var'] = ((a - am) * data['x'] + (b - bm)) ** 2
        var = var + data['var'].mean()

    return var / runs

## Final computations

Datasize = 2
final = pd.DataFrame(columns=['bias', 'var'], index=list('abcde'))
## ftype can have 5 values - (a)b (b)ax (c)ax+b (d)ax^2 and (e)ax^2+b

for ftype in list('abcde'):
    print(ftype)
    output = gbar(Datasize, ftype)
    bias = cal_bias(output, ftype)
    var = cal_var(output, Datasize, ftype)
    final.loc[ftype] = [bias, var]

final['error'] = final['bias'] + final['var']

## Checking out some github credentials