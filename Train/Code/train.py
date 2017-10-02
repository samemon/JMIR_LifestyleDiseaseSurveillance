'''
Description: This piece of code is train the model
and save those models for testing
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

'''
Description: Given numpy array x_train and y_train, this
function will train a linear regression model
REQUIRES:
- regularization: flag which will tell whether we should regularize or not
- loss function: This can be MAE,MSE,R
- cross_validation: flag which will tell if we should do cross-validation
- folds: number of folds for cross-validation
- x_train: a loaded numpy design matrix
- y_train: a loaded numpy label array
RETURNS:
This function basically returns a lasso regression model to be saved for later
and it also returns scores to be used for cross-validation
'''

def train_LASSOmodel(x_train,y_train,loss,regularization=True,
                cross_validation=True,folds=10,
                ):
    if(loss == 0):
        loss = 'neg_mean_absolute_error'
    elif(loss == 1):
        loss = 'neg_mean_squared_error'
    elif(loss == 2):
        loss = 'r2'
    model = linear_model.Lasso(random_state=0)
    #First we need to find a regularization hyper-parameter
    alphas = np.arange(0.001,1,0.001)
    scores = list()
    scores_std = list()
    counter = 0
    for alpha in alphas:
        model.alpha = alpha
        this_scores = cross_val_score(model, x_train, y_train,
                                      cv = folds, n_jobs = 1,
                                      scoring = loss)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))
        counter += 1
    opt = max(scores)
    print(scores)
    
    indexLst = [i for i,j in enumerate(scores) if j == opt]
    alphaIndex = indexLst[0]
    optAlpha = alphas[alphaIndex]
    print("Optimal Alpha:"+str(optAlpha))
    print ("score == " + str(max(scores)))
    #So now that we have figured out the alpha
    model = linear_model.Lasso(alpha=optAlpha)
    model.fit(x_train,y_train)
    return model    


if __name__ == "__main__":
    argv = sys.argv[1:]
    if(len(argv) == 5):
        x_train = np.load(argv[0])
        y_train = np.load(argv[1])
        folds = int(argv[2])
        loss = int(argv[3])
        fname = str(argv[4])
        model = train_LASSOmodel(x_train,y_train,loss,regularization=True,
                cross_validation=True,folds=10)
        with open(fname,'wb') as fid:
            cPickle.dump(model,fid)
        '/Users/samemon/desktop/JMIR_Lifestyle_Disease_Surveillance/Train/Models/train_model.pkl'
    else:
        print("Usage: \npython train.py <path to x_train.npy>"\
              "<path to y_train.npy> <no of folds of CV> <loss> <fname>\n"\
              "where loss = 0 if MAE, 1 if MSE and 2 if r2\n"\
              "fname is path+name+.pkl for model to be saved")
    
   
