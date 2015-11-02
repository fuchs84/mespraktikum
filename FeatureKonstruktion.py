import numpy as np
from scipy.signal import argrelmin, argrelmax
import math
from scipy import signal


__author__ = 'Sebastian'


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
            summand = 0
            for i in range(1, Windowsize):
                summand += dataMatrix[k-i,Sensors]
                summand += dataMatrix[k+i,Sensors]
            summand += dataMatrix[k,Sensors]
            array.append(summand/(2*Windowsize+1))
        median.append(array)
    return np.c_[dataMatrix[:, :2], median]

def Ableitung(dataMatrix, Sensor, Windowsize=1):
    Array = []
    for Sensors in Sensor:
        array = []
        for k in range(0, len(dataMatrix)-10):
            summand = 0

            summand += abs(dataMatrix[k+10,Sensors])
            summand = summand - abs(dataMatrix[k,Sensors])
            array.append(abs(summand))
        Array.append(array)
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