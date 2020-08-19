import numpy as np
import requests


def fetchdata():
    with open("in.dta", "x") as f_in:
        request_in = requests.get("http://work.caltech.edu/data/in.dta")
        f_in.write(request_in.text)
    with open("out.dta", "x") as f_out:
        request_out = requests.get("http://work.caltech.edu/data/in.dta")
        f_out.write(request_out.text)


## Executing code

#fetchdata()
data_train = np.loadtxt("in.dta")
data_test = np.loadtxt("out.dta")
