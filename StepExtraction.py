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

def getminimas(dataMatrix, Sensor=[290]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal)
    print maxAbsFreq
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)

    return argrelmin(Filtered[:,Sensor],order=25)

def getmaximas(dataMatrix, Sensor=[290]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal)
    print maxAbsFreq
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)

    return argrelmax(Filtered[:,Sensor],order=25)

def stepDetectionbackmiddle(dataMatrix):
    minimas = getminimas(dataMatrix)
    maxima = minimas[0]

    newmatrix = dataMatrix
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    for j in range(0,len(maxima)-1):
        middle = ((maxima[j]+maxima[j+1])/2)
        for k in range(maxima[j],middle):
            newmatrix[k,0]= maxima[j]
        for l in range(middle,maxima[j+1]):
            newmatrix[l,0] = maxima[j+1]
    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]


    return np.c_[dataMatrix,  newmatrix[:,0]]


def stepDetectionback(dataMatrix):
    minimas = getmaximas(dataMatrix)
    maxima = minimas[0]

    newmatrix = dataMatrix
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    for j in range(0,len(maxima)-1):

        for k in range(maxima[j],maxima[j+1]):
            newmatrix[k,0]= maxima[j]

    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]


    return np.c_[dataMatrix,  newmatrix[:,0]]

def stepDetectionTwoFoot(dataMatrix):
    left = getmaximas(dataMatrix,Sensor=[224])
    print left[0]
    right = getmaximas(dataMatrix,Sensor=[290])
    print right[0]
    maxima= np.concatenate((left[0],right[0]),axis=0)
    maxima = sorted(maxima)


    newmatrix = dataMatrix
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    for j in range(0,len(maxima)-1):
        middle = ((maxima[j]+maxima[j+1])/2)
        for k in range(maxima[j],middle):
            newmatrix[k,0]= maxima[j]
        for l in range(middle,maxima[j+1]):
            newmatrix[l,0] = maxima[j+1]
    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]
    return np.c_[dataMatrix,  newmatrix[:,0]]
