from scipy.signal import argrelmin,argrelmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeatureKonstruktion
import FourierTransformation
import Init
import pandas as pd
import math
import pylab as plb

__author__ = 'Sebastian'

def getminimas(dataMatrix, Sensor=[290]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal[0:13000])
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)
    print maxAbsFreq,maxAbsValue
    plt.plot(signal)
    plt.show()

    return argrelmin(Filtered[:,Sensor],order=25)

def getmaximas(dataMatrix, Sensor=[290]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal[0:13000])
    Filtered = FeatureKonstruktion.filter(dataMatrix,Sensor,maxAbsFreq)
    plt.plot(Filtered[:,Sensor],label="z-Acceleration Foot")
    plt.title("Filtered acceleration")
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("m/s^2")
    plt.show()
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
    Steps = []
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    Steps.append([0,maxima[0]])

    for j in range(0,len(maxima)-1):

        for k in range(maxima[j],maxima[j+1]):
            newmatrix[k,0]= maxima[j]
        Steps.append([maxima[j],maxima[j+1]])

    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]
    Steps.append([maxima[len(maxima)-1],len(dataMatrix[:,0])])
    Steparray = np.array(Steps)

    return Steparray, np.c_[dataMatrix,  newmatrix[:,0]]

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


def stepDetectionleft(dataMatrix):
    minimas = getmaximas(dataMatrix,Sensor=[202])
    maxima = minimas[0]

    newmatrix = dataMatrix
    Steps = []
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    Steps.append([0,maxima[0]])

    for j in range(0,len(maxima)-1):

        for k in range(maxima[j],maxima[j+1]):
            newmatrix[k,0]= maxima[j]
        Steps.append([maxima[j],maxima[j+1]])

    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]
    Steps.append([maxima[len(maxima)-1],len(dataMatrix[:,0])])
    extractedstep = np.array(Steps)
    return extractedstep, np.c_[dataMatrix,  newmatrix[:,0]]

def stepDetectionright(dataMatrix):
    minimas = getmaximas(dataMatrix,Sensor=[268])
    maxima = minimas[0]

    newmatrix = dataMatrix
    Steps = []
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    Steps.append([0,maxima[0]])

    for j in range(0,len(maxima)-1):

        for k in range(maxima[j],maxima[j+1]):
            newmatrix[k,0]= maxima[j]
        Steps.append([maxima[j],maxima[j+1]])

    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]
    Steps.append([maxima[len(maxima)-1],len(dataMatrix[:,0])])
    extractedstep = np.array(Steps)
    return extractedstep, np.c_[dataMatrix,  newmatrix[:,0]]

def videosteps(dataMatrix, elan):
    newMatrix= dataMatrix
    newlan =elan
    elan = pd.DataFrame(elan[:,1:],index=elan[:,0])
    elan = np.array(elan.ix['1Passgang'])
    print len(elan)
    print elan
    elan[:,:2] = elan[:,:2]/10
    elanhalbe = elan[:,:2]/2
    stepright, noneed = stepDetectionright(newMatrix)
    stepleft, noneed = stepDetectionleft(newMatrix)
    steps = np.concatenate((stepright,stepleft))
    realsteps = []
    for i in range(0,len(elan)):
        nearest =  find_nearest(steps[:,0],elanhalbe[i,0])
        nearest2 = steps[nearest,0]
        nearest = steps[nearest,1]
        realsteps.append([int(nearest2),int(nearest),int(elan[i,0])*20,int(elan[i,1])*20])
    realsteps = np.array(realsteps)

    plt.plot(newMatrix[:,202],label = "z-Vektor")

    plt.axvline(3,color ='r',label="hillclimber step")
    plt.axvline(3,color ='g',label="videolabeled step")
    plt.legend()
    plt.title("Stepextraction")
    plt.xlabel("Samples")

    for  xs in elanhalbe[:,0]:
        plt.axvline(x=xs,color = 'r')
    for  xs in elanhalbe[:,1]:
        plt.axvline(x=xs,color = 'r')
    for  xs in realsteps[:,0]:
        plt.axvline(x=xs,color = 'g')
    for  xs in realsteps[:,1]:
        plt.axvline(x=xs,color = 'g')
    #for  xs in stepleft[:,0]:
     #   plt.axvline(x=xs,color = 'g')
    #for  xs in stepleft[:,1]:
     #   plt.axvline(x=xs,color = 'g')
    plt.show()
    return newMatrix, realsteps


def findsync(dataMatrix):

    datamatrix = Init.getData(dataMatrix,sensors=["STE"],datas=[ "acc"])
    signal= dataMatrix[:,2]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal)
    Filtered = FeatureKonstruktion.filter(datamatrix,[2,3,4],maxAbsFreq)
    plt.plot(Filtered[:,2:])
    plt.show()


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx