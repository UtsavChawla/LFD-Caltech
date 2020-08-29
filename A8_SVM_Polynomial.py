import numpy as np
import requests

def fetchdata():
    with open("train.dta", "x") as f_in:
        request_in = requests.get("http://www.amlbook.com/data/zip/features.train")
        f_in.write(request_in.text)
    with open("test.dta", "x") as f_out:
        request_out = requests.get("http://www.amlbook.com/data/zip/features.test")
        f_out.write(request_out.text)

##Execution
#fetchdata()
data_train = np.loadtxt("train.dta")
data_test = np.loadtxt("test.dta")