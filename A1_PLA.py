import matplotlib.pyplot as plt
import random as rd
import numpy as np

# Defining a function to check convergence
def convergence(p0, p1, p2, setpos, setneg, xi, yi):
    for i in setpos:
        if ((p0) + (p1*xi[i]) + (p2*yi[i])) <= 0:
            return False
    for i in setneg:
        if ((p0) + (p1*xi[i]) + (p2*yi[i])) > 0:
            return False
    return True

def perceptron():
    # Step 1 : Creating input space
    xi = list()
    yi = list()
    for i in range(10):
        xi.append(rd.uniform(-1, 1))
        yi.append(rd.uniform(-1, 1))

    # Step 2 : Defining function f (y+ax+b = 0)
    xf = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
    yf = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
    coefficients = np.polyfit(xf, yf, 1)
    a= -coefficients[0]
    b= -coefficients[1]

    # Step 3 : classifying values according to f
    setpos = list()
    setneg = list()
    for i in range(10):
        if (yi[i] + (a*xi[i]) + b) <= 0 :
            setneg.append(i)
        else:
            setpos.append(i)

    # Step 4 : Finding function g (w0+w1x+w2y = 0) using PLA
    w0 = 0
    w1 = 0
    w2 = 0

    num = 0
    while(convergence(w0, w1, w2, setpos, setneg, xi, yi ) == False):
        index = rd.randint(0, 9)
        num = num + 1
        if (index in setpos) and ((w0 + (w1*xi[index]) + (w2*yi[index])) <= 0) :
            w0 = w0 + 1
            w1 = w1 + xi[index]
            w2 = w2 + yi[index]
            continue
        if (index in setneg) and ((w0 + (w1 * xi[index]) + (w2 * yi[index])) > 0):
            w0 = w0 - 1
            w1 = w1 - xi[index]
            w2 = w2 - yi[index]
            continue
        continue

    return num

rns = 0
for rnup in range(1000):
    print(rnup)
    rns = rns + perceptron()






# Appendix 1: Plotting input function f
polynomial = np.poly1d(coefficients)
x_axis = np.linspace(-1,1,100)
y_axis = polynomial(x_axis)
plt.plot(x_axis, y_axis)
for i in range(10):
    if i in setpos:
        plt.plot(xi[i], yi[i], 'go')
    else:
        plt.plot(xi[i], yi[i], 'ro')

coeg = coefficients.copy()
coeg[0] = -1 * (w1/w2)
coeg[1] = -1 * (w0/w2)
polynomial = np.poly1d(coeg)
x_axis = np.linspace(-1,1,100)
y_axis = polynomial(x_axis)
plt.plot(x_axis, y_axis)


