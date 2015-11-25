import numpy as np
from scipy.signal import argrelmin, argrelmax
import math
import matplotlib.pyplot as plt
import pandas as pd
import StepExtraction
import  Init
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
    for Sensors in Sensor:
        array = []
        for k in range(0, len(dataMatrix)-Windowsize):
            summand = 0

            summand += abs(dataMatrix[k+Windowsize,Sensors])
            summand = summand - abs(dataMatrix[k,Sensors])
            array.append(abs(summand))
        for k in range(len(dataMatrix)-Windowsize,len(dataMatrix)):
            array.append(dataMatrix[k,Sensors])
        Array.append(array)
    Array=np.array(Array)
    Array=np.array(Array.T)




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





def dotproduct(v1, v2):
  return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
def length(v):
  return math.sqrt(dotproduct(v, v))
def angle1(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def angle(vec1,vec2):
    nframe = np.zeros(shape=(len(vec1),7))
    df = pd.DataFrame(nframe)
    #nerechnung des winkels der beiden vektoren und speichern der vektoren in der matrix um sie auszugeben
    for a in range(0, len(vec1)):
        df.iloc[a,0] = math.degrees(angle1(vec1[a,:], vec2[a,:]))
        df.iloc[a,1:4] = vec1[a,:]
        df.iloc[a,4:7] = vec2[a,:]
    df = np.matrix(df)
    return df


#Fehlerfeatures der verschiedenen Fehler
#Fehler der Kategorie 1
def armstreckung(matrixnew):
    re1 = Init.getData(matrixnew,sensors=["RUF","RLL"],datas=[ "rE1","rE2","rE3"])
    anlematrixre1 = angle(re1[:,2:5],re1[:,5:8])
    anglearm = anlematrixre1[:,0]
    anlematrixre1 = angle(re1[:,8:11],re1[:,11:14])
    anglearm2 = anlematrixre1[:,0]
    anlematrixre1 = angle(re1[:,14:17],re1[:,17:20])
    anglearm3 = anlematrixre1[:,0]
    anglecomb= [ ]
    extractedstep,steps = StepExtraction.stepDetectionback(matrixnew)
    for i in range(0,len(anglearm)):
        anglecomb.append(np.sqrt((np.square(anglearm[i,0])+np.square(anglearm2[i,0])+np.square(anglearm3[i,0]))))

    print anglecomb
    maxanglearm = []
    minanglearm = []
    print anglearm
    for i in range(0,len(extractedstep)):
        maxpos = (np.argmax(anglecomb[(extractedstep[i,0]):(extractedstep[i,1])]))
        maxanglearm.append((anglecomb[(extractedstep[i,0])+maxpos]))
        minpos= (np.argmin(anglecomb[(extractedstep[i,0]):(extractedstep[i,1])]))
        minanglearm.append(anglecomb[(extractedstep[i,0])+minpos])

    angleE = np.c_[minanglearm,maxanglearm]
    return angleE


def Passgang(matrixnew):
    ACC = Init.getData(matrixnew,sensors=["RNS","LLL"],datas=["acc"],specifiedDatas="x")
    ACC2 = Init.getData(matrixnew,sensors=["LNS","RLL"],datas=["acc"],specifiedDatas="x")
    extractedstep,steps = StepExtraction.stepDetectionback(matrixnew)
    ACC = Ableitung(ACC[:,:],[2,3],1)[:,2:]
    ACC2 = Ableitung(ACC2[:,:],[2,3],1)[:,2:]
    zeitverschiebung =[]
    zeitverschiebunglinks =[]
    for i in range(0,len(extractedstep)):
        maxpos = (np.argmax(ACC2[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        minpos= (np.argmax(ACC2[(extractedstep[i,0]):(extractedstep[i,1]),1]))
        zeitverschiebung.append(abs(maxpos-minpos))
        maxpos = (np.argmax(ACC[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        minpos= (np.argmax(ACC[(extractedstep[i,0]):(extractedstep[i,1]),1]))
        zeitverschiebunglinks.append(abs(maxpos-minpos))


    vereinigung=[]
    for j in range(0,len(zeitverschiebunglinks)):
        if(zeitverschiebung[j]>=zeitverschiebunglinks[j]):
            vereinigung.append(zeitverschiebunglinks[j])
        else: vereinigung.append(zeitverschiebung[j])
    return vereinigung

def Stockaufsatz(matrixnew,Einheitsvektor ="rE1"):
    re1 = Init.getData(matrixnew,sensors=["RNS","LNS"],datas=[ Einheitsvektor])
    ACC = Init.getData(matrixnew,sensors=["RNS","LNS"],datas=["acc"],specifiedDatas="x")
    ACCright =  Ableitung(ACC[:,:],[2],1)[:,2:]
    ACCleft = Ableitung(ACC[:,:],[3],1)[:,2:]


    Vektor = np.zeros((len(re1),3))
    Vektor[:,2] = 1
    anlematrixre1 = angle(re1[:,2:5],Vektor)
    anlematrixre = angle(re1[:,5:8],Vektor)
    anglestickright = anlematrixre1[:,0]
    anglestickleft = anlematrixre[:,0]
    extractedstep,steps = StepExtraction.stepDetectionback(matrixnew)
    rightpeak = []
    leftpeak = []
    for i in range(0,len(extractedstep)):
        posleft = (np.argmax(ACCleft[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        leftpeak.append(anglestickleft[(extractedstep[i,0])+posleft,0])
        posright = (np.argmax(ACCright[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        rightpeak.append(anglestickright[(extractedstep[i,0])+posright,0])


    print rightpeak
    plt.subplot(2,1,1)
    plt.plot(rightpeak)

    plt.subplot(2,1,2)
    plt.plot(leftpeak)
    plt.show()
    return rightpeak,leftpeak




