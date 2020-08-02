import numpy as np
import pandas as pd
import math as mt


def sindata(N):
    # Row 0 is x and Row 1 is y
    data = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(N, 2)), columns=list('xy'))
    data['y'] = data.apply(lambda row: mt.sin(3.14 * row['x']), axis=1)
    return data


def learn(data):
    a = (data['x'] * data['y']).sum() / (data['x'] ** 2).sum()
    return a


def gbar():
    a = 0
    N = 2
    run = 1000
    for i in range(run):
        a = a + learn(sindata(N))

    return a / run


def bias(a):
    bias = 0
    run = 100
    ip = 2 / run
    x = -1
    for i in range(run):
        x = x + ip
        bias = bias + ((mt.sin(3.14 * x) - (a * x)) ** 2)

    return bias / run


def var(a):
    N = 2
    runs = 10000
    var = 0
    for i in range(runs):
        print(i)
        data = sindata(N)
        k = learn(data)
        var = var + (abs(a - k) * (data['x'] ** 2).sum() / N)

    return var / runs


## f(x) = sin(Pi*x) and g(x) = ax --> getting a
a = gbar()
bias = bias(a)
var = var(a)
