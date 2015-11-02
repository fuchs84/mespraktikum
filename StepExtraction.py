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


def getmaximas(dataMatrix, Sensor=[6]):
    plt.subplot(2,1,1)
    plt.plot(dataMatrix[5000:10000,Sensor])
    dataMatrix = FeatureKonstruktion.filter(dataMatrix,Sensor,5)
    plt.subplot(2,1,2)
    plt.plot(dataMatrix[5000:10000,Sensor])
    plt.show()
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(dataMatrix[5000:10000,Sensor])
    print maxAbsFreq
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)
    print argrelmax(Filtered[:,Sensor],order=10)
    return argrelmax(Filtered[:,Sensor],order=10)

def stepDetection(dataMatrix):
    maxima = getmaximas(dataMatrix)
    print maxima
    newmatrix = dataMatrix
    for i in range(0,maxima[0][0]):
        newmatrix[i,0]= maxima[0][0]
    for j in range(1,len(maxima[0])-1):
        middle = ((maxima[0][j]+maxima[0][j+1])/2)
        for k in range(maxima[0][j],middle):
            newmatrix[k,0]= maxima[0][j]
        for l in range(middle,maxima[0][j+1]):
            newmatrix[l,0] = maxima[0][j+1]
    for z in range(maxima[0][len(maxima[0])-1],len(dataMatrix[:,0])-1):
        newmatrix[z,0] = maxima[0][len(maxima[0])-1]


    return np.c_[dataMatrix,  newmatrix[:,0]]

