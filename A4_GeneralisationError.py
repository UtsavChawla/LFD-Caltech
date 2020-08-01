import math as mt
import numpy as np
import matplotlib.pyplot as plt

## Writting epsilon = f(N)

def VC(N, Dvc, delta):
    mh = (2 * N) ** Dvc
    eps = mt.sqrt((8 / N) * mt.log(4 * mh / delta))
    return eps


def RPB(N, Dvc, delta):
    mh = N ** Dvc
    eps = mt.sqrt((2 / N) * mt.log(2 * N * mh)) + mt.sqrt((2 / N) * mt.log(1 / delta)) + (1 / delta)
    return eps


def PVD(N, Dvc, delta):
    mh = (2 * N) ** Dvc

    eps = 0.5
    iter = 500
    for i in range(iter):
        eps = mt.sqrt((1 / N) * ((2 * eps) + mt.log(6 * mh / delta)))

    return eps


def DRY(N, Dvc, delta):
    eps = 0.5
    iter = 500
    for i in range(iter):
         eps = mt.sqrt((1 / (2 * N)) * ((4 * eps * (1 + eps)) + mt.log(4 / delta) + (2 * Dvc * mt.log(N))))

    return eps


## Calculating error bounds
delta = 0.05
Dvc = 50
N = 10

vc = np.array([VC(n, Dvc, delta) for n in range(3, N)])
rpb = np.array([RPB(n, Dvc, delta) for n in range(3, N)])
pvd = np.array([PVD(n, Dvc, delta) for n in range(3, N)])
dry = np.array([DRY(n, Dvc, delta) for n in range(3, N)])

## Plotting error bounds
plt.plot(vc, label='Original VC')
plt.plot(rpb, label='Rademacher Penalty Bound')
plt.plot(pvd, label='Parrondo and Van den Broek')
plt.plot(dry, label='Devroye')
plt.legend()