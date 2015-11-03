from scipy.signal import argrelmin,argrelmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeatureKonstruktion
import FourierTransformation
import pandas as pd
import math
import pylab as plb

__author__ = 'Sebastian'

def getminimas(dataMatrix, Sensor=[179]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal)
    print maxAbsFreq
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)
    return argrelmin(Filtered[:,Sensor],order=10)

def getmaximas(dataMatrix, Sensor=[179]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal)
    print maxAbsFreq
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)
    return argrelmax(Filtered[:,Sensor],order=10)

def stepDetection(dataMatrix):
    maximas = getmaximas(dataMatrix)
    print maximas[0]
    minimas = getminimas(dataMatrix)
    print minimas[0]
    maxima= np.concatenate((maximas[0],minimas[0]),axis=0)
    maxima = sorted(maxima)
    newmatrix = dataMatrix
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    for j in range(1,len(maxima)-1):
        middle = ((maxima[j]+maxima[j+1])/2)
        for k in range(maxima[j],middle):
            newmatrix[k,0]= maxima[j]
        for l in range(middle,maxima[j+1]):
            newmatrix[l,0] = maxima[j+1]
    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])-1):
        newmatrix[z,0] = maxima[len(maxima)-1]


    return np.c_[dataMatrix,  newmatrix[:,0]]

