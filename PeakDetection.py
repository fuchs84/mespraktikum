__author__ = 'Sebastian'

from scipy.signal import argrelmin,argrelmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pylab as plb

#gibt durchscnittliche Zeit zwischen den  Peaks im Graphen an
def timebetweenpeaks(dataMatrix, Sensor,Min = False):
    timedifference = []
    if Min==False:
        for Sensors in Sensor:
            peaks = (argrelmax(dataMatrix[:,Sensors],order=10))
            distance = []
            for p in range(0,len(peaks[0])-1):
                distance.append((peaks[0][p+1]-peaks[0][p]))
            number=0
            for d in distance:
                number +=d
            timedifference.append((number/len(distance)))

    else:
        for Sensors in Sensor:
            peaks = (argrelmax(dataMatrix[:,Sensors],order=10))
            distance = []
            for p in range(0,len(peaks[0])-1):
                distance.append((peaks[0][p+1]-peaks[0][p]))
            number=0
            for d in distance:
                number +=d
            timedifference.append((number/len(distance)))

    return pd.DataFrame(timedifference)



def Steigung(dataMatrix, Sensor, Windowsize=1):
    tobi = []
    for Sensors in Sensor:
        array = []
        for k in range(0, len(dataMatrix)-10):
            summand = 0

            summand += abs(dataMatrix[k+10,Sensors])
            summand = summand - abs(dataMatrix[k,Sensors])
            array.append(abs(summand))
        tobi.append(array)
    return tobi

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
    return median

#Gibt die Autocorrelation eines Sensorwertes oder Vectors an
def autocorr(dataMatrix,Sensor):
    autocorrelation = []
    for Sensors in Sensor:
        result = np.correlate(dataMatrix[:,Sensors], dataMatrix[:,Sensors], mode='full')
        autocorrelation.append(result[result.size/2:])
    return autocorrelation

#berechnet die Standardabweichung eines oder mehrerer Sensoren
def getdeviation(dataMatrix,Sensor):
    deviation = []

    for Sensors in Sensor:
        mean = getmean(dataMatrix,[Sensors])
        arrayused = 0
        for i in range(0,len(dataMatrix)):
            arrayused += math.pow((dataMatrix[i,Sensors]-mean),2)
        arrayused=arrayused/len(dataMatrix)
        deviation.append(arrayused)
    return deviation


def getmean(dataMatrix, Sensor):
    means = []
    for Sensors in Sensor:
        arrayused = 0
        for i in range(0,len(dataMatrix)):
            arrayused += dataMatrix[i,Sensors]
        arrayused=arrayused/len(dataMatrix)
        means.append(arrayused)
    return means



def getrelmaximas(dataMatrix, Sensor,Min = False):
    extrema = []
    if Min==False:
        for Sensors in Sensor:
            extrema.append(dataMatrix[argrelmax(dataMatrix[:,Sensors],order=10),Sensors][0])
    else:
        for Sensors in Sensor:
            extrema.append(dataMatrix[argrelmin(dataMatrix[:,Sensors],order=10),Sensors][0])
    return pd.DataFrame.transpose(pd.DataFrame(extrema))

def histogramm(data):
    hist = []
    for tests in data[0]:
        hist.append(np.histogram(tests))
    return pd.DataFrame(hist)





#---------------------------------------------------------veraltet----------------------------------------------------------------
def vectoronquaternionposition(dataMatrix):
    #berechnen der Vektoren aus den Quaternions und speichern anstelle der Quaternions, Position 4 ist der addierte Wert der Vektoren
    traindata = dataMatrix
    for i in range(11,180,13):
        print i
        x,y,z = calculatevector(traindata, [i,i+1,i+2,i+3])
        traindata[:,i]=x[0]
        traindata[:,i+1]=x[1]
        traindata[:,i+2]=x[2]
        traindata[:,i+3]=x[0]+x[1]+x[1]
        print "quat calc"
        #filtern der Daten duch Medianfilter
        medi = medianfilter(traindata,[i,i+1,i+2,i+3],20)
        traindata[:len(medi[0]),i]=medi[0]
        traindata[:len(medi[0]),i+1]=medi[1]
        traindata[:len(medi[0]),i+2]=medi[2]
        traindata[:len(medi[0]),i+3]=medi[3]
        print "train med calc"
        diff = Steigung(traindata,[i+3])
        traindata[:len(diff[0]),i+3]=diff[0]
        print "train diff calc"
    np.savetxt('Vectoren.csv', fmt=['%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f'] ,X= traindata, delimiter='\t')
    print "saved"


def calculatevector(dataMatrix, Sensor, axis=1):
    nframe = np.zeros(shape=(len(dataMatrix),3))
    vec10= pd.DataFrame(nframe)
    vec11= pd.DataFrame(nframe)
    vec12= pd.DataFrame(nframe)
    for a in range(0, len(dataMatrix)):
            #Quaternion nach schema wq xq yq zq
            wq = dataMatrix[a,Sensor[0]]
            xq = dataMatrix[a,Sensor[1]]
            yq = dataMatrix[a,Sensor[2]]
            zq = dataMatrix[a,Sensor[3]]
            m11 = xq*xq+wq*wq-yq*yq-zq*zq
            m12 = 2*(xq*yq-wq*zq)
            m13 = 2*(xq*zq+wq*yq)
            m21 = 2*(wq*zq+xq*yq)
            m22 = wq*wq-xq*xq+yq*yq-zq*zq
            m23 = 2*(yq*zq-wq*xq)
            m31 = 2*(xq*zq-wq*yq)
            m32 = 2*(wq*xq+yq*zq)
            m33 = wq*wq-xq*xq-yq+yq+zq*zq
            c = np.matrix([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
            e1= np.matrix('1; 0; 0')
            e2= np.matrix('0; 1; 0')
            e3= np.matrix('0; 0; 1')
            vec10.iloc[a,0] =(c*e1)[0]
            vec10.iloc[a,1] =(c*e1)[1]
            vec10.iloc[a,2] =(c*e1)[2]
            vec11.iloc[a,0] = (c*e2)[0]
            vec11.iloc[a,1] = (c*e2)[1]
            vec11.iloc[a,2] = (c*e2)[2]

            vec12.iloc[a,0] = (c*e3)[0]
            vec12.iloc[a,1] = (c*e3)[1]
            vec12.iloc[a,2] = (c*e3)[2]
    return vec10,vec11,vec12

#gibt durchscnittliche Zeit zwischen den  Peaks im Graphen an
def timebetweenpeaks(dataMatrix, Sensor,Min = False):
    timedifference = []
    if Min==False:
        for Sensors in Sensor:
            peaks = (argrelmax(dataMatrix[:,Sensors],order=10))
            distance = []
            for p in range(0,len(peaks[0])-1):
                distance.append((peaks[0][p+1]-peaks[0][p]))
            number=0
            for d in distance:
                number +=d
            timedifference.append((number/len(distance)))

    else:
        for Sensors in Sensor:
            peaks = (argrelmax(dataMatrix[:,Sensors],order=10))
            distance = []
            for p in range(0,len(peaks[0])-1):
                distance.append((peaks[0][p+1]-peaks[0][p]))
            number=0
            for d in distance:
                number +=d
            timedifference.append((number/len(distance)))

    return pd.DataFrame(timedifference)

def fourier(dataMatrix, Sensor):
    fftall = []
    for Sensors in Sensor:
        arrayused = dataMatrix[:,Sensors]
        fftsensor = np.fft.fft(arrayused)
        fftall.append(fftsensor)
    return fftall
