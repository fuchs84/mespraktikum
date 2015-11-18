import numpy as np
from scipy.signal import argrelmin, argrelmax
import math
import matplotlib.pyplot as plt
import pandas as pd
import StepExtraction
from scipy import signal


__author__ = 'Sebastian'


def histogramStride(dataMatrix):
    matrixnew = dataMatrix
    Steparray, step = StepExtraction.stepDetectionback(matrixnew)
    step1 = pd.DataFrame(step)
    step1 = step1.iloc[:,310]
    step1=  np.array(step1.value_counts())

    plt.hist(step1)
    plt.show()

def stepvariance(dataMatrix, stepLabel):
    variance = np.zeros(shape=(len(stepLabel),len(dataMatrix[1,:])))
    for i in range(0, len(stepLabel)):
        temparray = np.array(dataMatrix[stepLabel[i,0]:stepLabel[i,1],:])
        variance [i,[0,1]] = temparray[5,[0,1]]
        for j in range(2,len(dataMatrix[1,:])):
            variance[i,j] = np.var(temparray[:,j])
    return variance

def stepmean(dataMatrix, stepLabel):
    mean = np.zeros(shape=(len(stepLabel),len(dataMatrix[1,:])))
    for i in range(0, len(stepLabel)):
        temparray = np.array(dataMatrix[stepLabel[i,0]:stepLabel[i,1],:])
        mean [i,[0,1]] = temparray[0,[0,1]]
        for j in range(2,len(dataMatrix[1,:])):
            mean[i,j] = np.mean(temparray[:,j])
    return mean

def stepmedian(dataMatrix, stepLabel):
    median = np.zeros(shape=(len(stepLabel),len(dataMatrix[1,:])))
    for i in range(0, len(stepLabel)):
        temparray = np.array(dataMatrix[stepLabel[i,0]:stepLabel[i,1],:])
        median [i,[0,1]] = temparray[0,[0,1]]
        for j in range(2,len(dataMatrix[1,:])):
            median[i,j] = np.median(temparray[:,j])
    return median

def stepaverage(dataMatrix, stepLabel):
    average = np.zeros(shape=(len(stepLabel),len(dataMatrix[1,:])))
    for i in range(0, len(stepLabel)):
        temparray = np.array(dataMatrix[stepLabel[i,0]:stepLabel[i,1],:])
        average [i,[0,1]] = temparray[0,[0,1]]
        for j in range(2,len(dataMatrix[1,:])):
            average[i,j] = np.average(temparray[:,j])
    return average




def filter(dataMatrix,Sensors, highcut=0, lowcut= 0, Ordnung =2, filtertype = "lowpass"):
    fs = 50
    nyquist = 0.5*50
    cutoffhigh= highcut/nyquist
    cutofflow = lowcut/nyquist
    if filtertype == "lowpass":
        bandbreite = cutoffhigh
    elif filtertype == "highpass":
        bandbreite = cutofflow
    elif filtertype == "bandpass":
        bandbreite = [cutofflow,cutoffhigh]

    y= np.array(dataMatrix)
    k = np.array(y, dtype=float)
    y2 =np.array(y)
    b, a = signal.butter(Ordnung, bandbreite, btype = filtertype)
    for Sensor in Sensors:
        y2[:,Sensor] = signal.lfilter(b, a, k[:,Sensor])  # standard filter
    y2 = np.array(y2)
    return np.c_[dataMatrix[:, :2],  y2[:,2:]]


def medianfilter(dataMatrix,Sensor, Windowsize):
    median = []
    for Sensors in Sensor:
        array = []
        for k in range(0, len(dataMatrix)-Windowsize):
            summand = []
            for i in range(1, Windowsize):
                summand.append(dataMatrix[k-i,Sensors])
                summand.append(dataMatrix[k+i,Sensors])
            summand.append(dataMatrix[k,Sensors])
            array.append(np.median(summand))
        for k in range(len(dataMatrix)-Windowsize,len(dataMatrix)):
            array.append(dataMatrix[k,Sensors])
        median.append(array)
    print dataMatrix[:,:2].shape
    median=np.array(median)
    median=np.array(median.T)
    print median.shape
    return np.c_[dataMatrix[:, :2], median]

def Ableitung(dataMatrix, Sensor, Windowsize=1):
    Array = []
    print Sensor
    for Sensors in Sensor:
        array = []
        for k in range(0, len(dataMatrix)-10):
            summand = 0

            summand += abs(dataMatrix[k+10,Sensors])
            summand = summand - abs(dataMatrix[k,Sensors])
            array.append(abs(summand))
        for k in range(len(dataMatrix)-Windowsize,len(dataMatrix)):
            array.append(dataMatrix[k,Sensors])
        Array.append(array)
    Array=np.array(Array)
    Array=np.array(Array.T)

    print dataMatrix[:,:2].shape
    print Array.shape


    return np.c_[dataMatrix[:, :2], Array]

#gibt durchscnittliche Zeit zwischen den  Peaks im Graphen an
def timebetweenpeaks(dataMatrix, Sensor,Min = False,order=10):
    timedifference = []
    if Min==False:
        for Sensors in Sensor:
            peaks = (argrelmax(dataMatrix[:,Sensors],order=order))
            distance = []
            for p in range(0,len(peaks[0])-1):
                distance.append((peaks[0][p+1]-peaks[0][p]))
            number=0
            for d in distance:
                number +=d
            timedifference.append((number/len(distance)))

    else:
        for Sensors in Sensor:
            peaks = (argrelmax(dataMatrix[:,Sensors],order=order))
            distance = []
            for p in range(0,len(peaks[0])-1):
                distance.append((peaks[0][p+1]-peaks[0][p]))
            number=0
            for d in distance:
                number +=d
            timedifference.append((number/len(distance)))

    return timedifference