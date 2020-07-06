import numpy as np
import random as rd

def cointoss(num_coin, num_toss):
    # Tossing coins
    head_count = np.random.binomial(num_toss, 0.5, num_coin)

    # Getting frequency of head for 3 cases - first, random and min
    v1 = head_count[0]
    vrand = rd.choice(head_count)
    vmin = np.min(head_count)

    return v1/10 , vrand/10 , vmin/10

# Tossing coin - generating frequency distribution
num_coin = 1000
num_toss = 10
num_iter = 100000
v = np.empty((num_iter,3), float)

for i in range(num_iter):
    print(i)
    output = cointoss(num_coin, num_toss)
    v[i] = output

# Checking for Hoeffding Inequality
u = v.copy()
v[:,0] = 0.5
v[:,1] = 0.5
v[:,2] = 0

delta = abs(np.subtract(v,u))

d1 = delta[:,0]
p1 = np.count_nonzero(d1>0.29)/100000

drand = delta[:,1]
prand = np.count_nonzero(drand>0.29)/100000

dmin = delta[:,2]
pmin = np.count_nonzero(dmin>0.29)