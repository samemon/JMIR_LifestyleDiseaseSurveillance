'''
Description: This piece of code is to test the model
on a provided test set
-------------------------------------
Author: Shahan A. Memon
Advisors: Ingmar Weber, Saquib Razak

Copyright: Carnegie Mellon University,
Qatar Computing Research Institute 2017
--------------------------------------
'''

import csv
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import random
import sys
import math
import cPickle

from sklearn import preprocessing
import matplotlib.pyplot as plt

def findMSE(true,pred):
    return np.mean((true-pred)**2)

def findCorr(true,pred):
    return np.corrcoef(true,pred)[[0],[1]][0]

def findSMAPE(true,pred):
    SMAPE = 0
    ratio = []
    for index in range(len(true)):
        sub = abs(true[index]-pred[index])
        avg = (true[index]+pred[index])/2.0
        ratio.append(sub*1.0/avg)
    SMAPE = sum(ratio)/len(true)*100.0
    return SMAPE

if __name__ == "__main__":
    argv = sys.argv[1:]
    print len(argv)
    print argv
    if(len(argv) == 3):
        x_test = np.load(argv[0])
        y_test = np.load(argv[1])
        model = linear_model.Lasso(random_state=0)
        with open(str(argv[2]),'rb') as fid:
            model = cPickle.load(fid)
        preds = np.array(model.predict(x_test))
        MSE = findMSE(y_test,preds)
        R = findCorr(y_test,preds)
        SMAPE = findSMAPE(y_test,preds)
        print("MSE ====== :" + str(MSE))
        print("R ====== :" + str(R))
        print("SMAPE ====== :" + str(SMAPE))
        halfPreds = np.split(preds,2)
        halfTrues = np.split(y_test,2)
        MSE = (findMSE(halfTrues[0],halfPreds[0]) +
               findMSE(halfTrues[1],halfPreds[1]))/2.0
        R = (findCorr(halfTrues[0],halfPreds[0]) +
               findCorr(halfTrues[1],halfPreds[1]))/2.0
        SMAPE = (findSMAPE(halfTrues[0],halfPreds[0]) +
               findSMAPE(halfTrues[1],halfPreds[1]))/2.0
        print("1/2MSE ====== :" + str(MSE))
        print("1/2R ====== :" + str(R))
        print("1/2SMAPE ====== :" + str(SMAPE))
        
    else:
        print("Usage: <fp-test_x> <fp-test_y> <fp-model>")
        
