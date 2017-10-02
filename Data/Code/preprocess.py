'''
Description: This piece of code is used to preprocess the
raw data from csv files to create numpy arrays for train
and test.
-------------------------------------
Author: Shahan A. Memon
Advisors: Ingmar Weber, Saquib Razak

Copyright: Carnegie Mellon University,
Qatar Computing Research Institute 2017
--------------------------------------
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
from sklearn import preprocessing

'''
Description: This function simply reads a file line by line
and creates a python list out of it. For our purposes it reads
the states or keywords for which we have data available.
Sometimes data for some states or keywords is missing,
and so we might not have data available for all the 52 states for example.
This is a simple list creator for a file in which data is separated by line
Requires:
- filename: file which consists of the states/keywords
for which data is available
Returns: A list of those states
'''

def readLst(filename):
    #Initialize an empty list
    lst = []
    with open(filename) as f:
        for line in f:
            #Read each line, remove whitespaces and
            #append to list
            lst.append(line.rstrip())

    #return sorted list
    return sorted(lst)

'''
This function reads the training and testing data
from a csv file to a numpy array
REQUIRES:
- states (list of states)
- fpcsv (filepath to the csv file)
NOTE: the csv file format should match our format
PLEASE LOOK into Data/Raw/ to see the file format
of our csv files
- trn_years (list of training years)
- tst_years (list of testing years)
- scale (whether to scale the data or not and how)
RETURNS:
A tuple of (features,x_train,y_train,x_test,y_test)
'''

def readCSV(states,fpcsv,trn_years,tst_years,scale):
    #Read entire data as a string array 
    all_data = np.genfromtxt(fpcsv,
                             dtype=None,
                             delimiter=',')
    '''
    Now we read the labels which are supposed to be in the 3rd col
    '''
    Y = all_data[:,[2]].flatten()[1:].astype(np.float)
    X = all_data[1:,3:].astype(np.float)
    if(scale == 1):
        X = X / X.max(axis=0)
    features = all_data[[0],3:].flatten()
    '''
    Now that we have X and Y, let's divide them into years
    #This code is assuming that years in the csv file are
    in ascending order
    '''
    year_to_arrayX = {}
    year_to_arrayY = {}
    numberOfYears = len(X[:,[0]])/len(states)
    assert(numberOfYears == len(X[:,[0]])*1.0/len(states))
    all_years = [2011+i for i in range(numberOfYears)]
    assert(set(trn_years) <= set(all_years))
    assert(set(tst_years) <= set(all_years))
    for i in range(numberOfYears):
        year_to_arrayX[all_years[i]] = X[i*len(states):(i+1)*len(states)]
        year_to_arrayY[all_years[i]] = Y[i*len(states):(i+1)*len(states)]
    #Random initialization
    x_train = year_to_arrayX[trn_years[0]]
    y_train = year_to_arrayY[trn_years[0]]
    for y in trn_years[1:]:
        x_train = np.concatenate((x_train,year_to_arrayX
                                  [y]), axis=0)
        y_train = np.concatenate((y_train,year_to_arrayY
                                  [y]), axis=0)
    #Random initialization
    x_test = year_to_arrayX[tst_years[0]]
    y_test = year_to_arrayY[tst_years[0]]
    for y in tst_years[1:]:
        x_test = np.concatenate((x_test,year_to_arrayX
                                  [y]), axis=0)
        y_test = np.concatenate((y_test,year_to_arrayY
                                  [y]), axis=0)
    if(scale == 2):
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    if(scale == 3):
        x_train = preprocessing.scale(x_train)
        x_test = preprocessing.scale(x_test)
    
    return (x_train,y_train,x_test,y_test,features)
    
    
if __name__ == "__main__":
    argv = sys.argv[1:]
    if(len(argv)== 6):
        assert(len(argv) == 6)
        fps = str(argv[0])
        fpcsv = str(argv[1])
        fpo = str(argv[2])
        trn_years = []
        tst_years = []
        scaleFlag = 1
        #Checking if file paths and dirs exist
        if(os.path.isdir(fpo) and os.path.isfile(fps) and
           os.path.isfile(fpcsv)):
            #list of train and test years
            try:
                trn_years =  map(int,str(argv[3]).split(","))
                tst_years =  map(int,str(argv[4]).split(","))
            except:
                print("Error reading training/testing years")
            try:
                scaleFlag = int(argv[5])
                assert(scaleFlag in [0,1,2,3])
            except:
                print("invalid scaling choice")
            states = readLst(fps)
            #readCSV will return a tuple of 4 numpy arrays
            x_train,y_train,x_test,y_test,features = readCSV(states,
                                                    fpcsv,
                                                    sorted(trn_years),
                                                    sorted(tst_years),
                                                    scaleFlag)
            #Saving the numpy arrays
            try:
                np.save(fpo+"/"+"x_train", x_train)
                np.save(fpo+"/"+"y_train", y_train)
                np.save(fpo+"/"+"x_test", x_test)
                np.save(fpo+"/"+"y_test", y_test)
                np.save(fpo+"/"+"features",features)
                print("Saved Arrays successfully in the folder:"+fpo)
            except:
                print("Error saving arrays")
        else:
            print("File/Directory Path Error")
        
    else:
        assert(len(argv) != 6)
        print("Error: This program needs 6 arguments\n"\
              "Usage: python preprocess.py <fp-s>"\
              "<fp-csv> <fp-o> <train-years> <test-years>"\
              "<scale-flag>\n"\
              "fp-s: file path to the list of states\n"\
              "fp-csv: file path to the csv file of data\n"\
              "fp-o: directory path to save numpy arrays\n"\
              "train-years: comma separated years, e.g. 2011,2012\n"\
              "test-years: comma separated years, e.g. 2013,2014\n"\
              "scale-flag:\n"\
              "- 0 for no normalization\n"\
              "- 1 for max normalization\n"\
              "- 2 for mean normalization\n"\
              "- 3 mean normalize test separately from train")
