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


#gibt den winkel zwischen zwei sensoren
def winkelgesamt(matrixnew,Sensoren= ["RUA","RLA"]):
    re1 = Init.getData(matrixnew,sensors=Sensoren,datas=[ "rE1","rE2","rE3"])
    print len(matrixnew)
    anlematrixre1 = angle(re1[:,11:14],re1[:,2:5])
    anglearm = anlematrixre1[:,0]
    anlematrixre1 = angle(re1[:,14:17],re1[:,5:8])
    anglearm2 = anlematrixre1[:,0]
    anlematrixre1 = angle(re1[:,17:20],re1[:,8:11])
    anglearm3 = anlematrixre1[:,0]
    anglecomb= [ ]
    plt.plot(anglearm)
    plt.plot(anglearm2)
    plt.plot(anglearm3)
    plt.show()
    for i in range(0,len(anglearm)):
        anglecomb.append(np.sqrt((np.square(anglearm2[i,0]))+(np.square(anglearm[i,0]))+(np.square(anglearm3[i,0]))))
    print len(anglecomb)
    return anglecomb


# gibt den das winkelmaximum und minimum eines sensorpaars innerhalb eines schrittes zurueck
def sensorwinkel(matrixnew, extractedstep, Sensoren= ["RUA","RLA"]):
    re1 = Init.getData(matrixnew,sensors=Sensoren,datas=[ "rE1","rE2","rE3"])
    anlematrixre1 = angle(re1[:,11:14],re1[:,2:5])
    anglearm = anlematrixre1[:,0]
    anlematrixre1 = angle(re1[:,14:17],re1[:,5:8])
    anglearm2 = anlematrixre1[:,0]
    anlematrixre1 = angle(re1[:,17:20],re1[:,8:11])
    anglearm3 = anlematrixre1[:,0]
    anglecomb= [ ]

    for i in range(0,len(anglearm)):
        anglecomb.append(np.sqrt((np.square(anglearm2[i,0]))+(np.square(anglearm[i,0]))+(np.square(anglearm3[i,0]))))

    print anglecomb
    maxanglearm = []
    minanglearm = []
    for i in range(0,len(extractedstep)):
        maxpos = (np.argmax(anglecomb[(extractedstep[i,0]):(extractedstep[i,1])]))
        maxanglearm.append((anglecomb[(extractedstep[i,0])+maxpos]))
        minpos= (np.argmin(anglecomb[(extractedstep[i,0]):(extractedstep[i,1])]))
        minanglearm.append(anglecomb[(extractedstep[i,0])+minpos])

    angleE = np.c_[minanglearm,maxanglearm]
    print "---------------------------------------------------------------------"
    angleE= 180-angleE
    return angleE


#Fehlerfeatures der verschiedenen Fehler
#Fehler der Kategorie 1
#schulterbewegung und armstreckung ist ein kombiniertes feature zur armverwendung
def schulterbewegung(matrixnew,extractedstep):
    angleschulterright = sensorwinkel(matrixnew,extractedstep,Sensoren=["STE","RUA"])
    angleschulterleft = sensorwinkel(matrixnew,extractedstep,Sensoren=["STE","LUA"])
    angleschulterright = abs(angleschulterright[:,0]-angleschulterright[:,1])
    angleschulterleft = abs(angleschulterleft[:,0]-angleschulterleft[:,1])
    print angleschulterleft
    return angleschulterleft,angleschulterright

def armstreckung(matrixnew,extractedstep):
    anglearmterright = sensorwinkel(matrixnew,extractedstep,Sensoren=["RLA","RUA"])
    anglearmlterleft = sensorwinkel(matrixnew,extractedstep,Sensoren=["LLA","LUA"])
    anglearmterright = abs(anglearmterright[:,0]-anglearmterright[:,1])
    anglearmlterleft = abs(anglearmlterleft[:,0]-anglearmlterleft[:,1])
    return anglearmlterleft,anglearmterright





#Erkennt zeitunterschied zwischen den Stock und Fussaufsatz
def Passgang(matrixnew,extractedstep):
    ACC = Init.getData(matrixnew,sensors=["RNS","LLL"],datas=["acc"],specifiedDatas="x")
    ACC2 = Init.getData(matrixnew,sensors=["LNS","RLL"],datas=["acc"],specifiedDatas="x")

    ACC = Ableitung(ACC[:,:],[2,3],1)[:,2:]
    ACC2 = Ableitung(ACC2[:,:],[2,3],1)[:,2:]
    zeitverschiebung =[]
    zeitverschiebunglinks =[]
    for i in range(0,len(extractedstep)):

        maxpos = (np.argmax(ACC2[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        minpos= (np.argmax(ACC2[(extractedstep[i,0]):(extractedstep[i,1]),1]))
        print abs(maxpos-minpos)
        print "_______"
        print (extractedstep[i,1]-extractedstep[i,0])
        zeitverschiebung.append(float(abs(maxpos-minpos))/float((extractedstep[i,1]-extractedstep[i,0])))
        maxpos = (np.argmax(ACC[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        minpos= (np.argmax(ACC[(extractedstep[i,0]):(extractedstep[i,1]),1]))
        zeitverschiebunglinks.append(float(abs(maxpos-minpos))/float((extractedstep[i,1]-extractedstep[i,0])))


    vereinigung=[]
    for j in range(0,len(zeitverschiebunglinks)):
        if(zeitverschiebung[j]>=zeitverschiebunglinks[j]):
            vereinigung.append(zeitverschiebunglinks[j])
        else: vereinigung.append(zeitverschiebung[j])
    plt.plot(vereinigung)
    plt.show()
    return vereinigung

#gibt den Winkel im Moment des stockaufsatzes( Annahme, wenn kleiner 180 wird der Stock nicht vor dem Koerper aufgesetzt
def Stockaufsatz(matrixnew,extractedstep,Einheitsvektor ="rE1"):
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

    rightpeak = []
    leftpeak = []
    for i in range(0,len(extractedstep)):
        posleft = (np.argmax(ACCleft[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        leftpeak.append(anglestickleft[(extractedstep[i,0])+posleft,0])
        posright = (np.argmax(ACCright[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        rightpeak.append(anglestickright[(extractedstep[i,0])+posright,0])


    print rightpeak

    return rightpeak,leftpeak


#Fehler 2. Art Muskulatur nicht optimal ausgelastst

#handverkrampft gibt den winkel zwischen stoch und untera
def handverkrampft(matrixnew,extractedstep):
    angleright = winkelgesamt(matrixnew,Sensoren=["RNS","RLA"])
    angleleft = winkelgesamt(matrixnew,Sensoren=["LNS","LLA"])
    varleft=[]
    meanleft=[]
    varright=[]
    meanright=[]
    for i in range(0,len(extractedstep)):
        rightstep = angleright[(extractedstep[i,0]):(extractedstep[i,1])]
        leftstep = angleleft[(extractedstep[i,0]):(extractedstep[i,1])]
        sumleft=0
        sumright =0
        for j in range(0,len(leftstep)):
            sumleft+= leftstep[j]
            sumright+= rightstep[j]
        meanleft.append(np.mean((sumleft/abs((extractedstep[i,0])-(extractedstep[i,1])),sumright/abs((extractedstep[i,0])-(extractedstep[i,1])))))
        sumleft =0
        sumright=0
        for k in range(0,len(leftstep)):
            sumleft += np.square((leftstep[k]-meanleft[i]))
            sumright += np.square((rightstep[k]-meanleft[i]))
        varleft.append((sumleft+sumright)/(2*abs((extractedstep[i,0])-(extractedstep[i,1]))))





    return meanleft,varleft

#Ueberpruefen des aufrechten winkels
def aufrechtgehen(matrixnew, extractedstep):
        back = Init.getData(matrixnew,sensors=["STE","CEN"],datas=[ "rE1"])
        Vektor = np.zeros((len(back),3))
        Vektor[:,0] = 1
        angleSTEmovement = angle(back[:,2:5],Vektor)
        angleCENmovement = angle(back[:,5:8],Vektor)
        Vektor[:,0] = 0
        Vektor[:,2] = 1
        angleSTEupwards = angle(back[:,2:5],Vektor)
        angleCENupwards = angle(back[:,5:8],Vektor)
        angleSTEupwardsstep = []
        angleSTEmovementstep =[]
        angleSTEmovementstep = []
        angleCENupwardsstep = []
        for i in range(0,len(extractedstep)):
            angleCENupwardsstep.append(np.sum(angleCENupwards[(extractedstep[i,0]):(extractedstep[i,1])])/((extractedstep[i,1])-(extractedstep[i,0])))
            angleSTEupwardsstep.append(np.sum(angleSTEupwards[(extractedstep[i,0]):(extractedstep[i,1])])/((extractedstep[i,1])-(extractedstep[i,0])))


        plt.subplot(2,1,1)
        plt.plot(angleSTEupwardsstep)
        plt.subplot(2,1,2)

        plt.plot(angleCENupwardsstep)
        plt.show()
        difference = (np.absolute([x-y for x,y in zip(angleCENupwardsstep,angleSTEupwardsstep)]))
        return  angleSTEupwardsstep,angleCENupwardsstep, difference



#Wenn der Fuss mit einem Winkel zu ca 90 grad aufgesetzt wir ist anzunehmen das abgerollt wird da dann mit der ferse aufgekommen wird
def Fussabrollen(matrixnew,extractedstep):
    ACC = Init.getData(matrixnew,sensors=["RLL","LLL"],datas=["acc"],specifiedDatas="x")
    ACCright =  Ableitung(ACC[:,:],[2],1)[:,2:]
    ACCleft = Ableitung(ACC[:,:],[3],1)[:,2:]




    anlematrixre1 = winkelgesamt(matrixnew,Sensoren=["RUA","RLA"])
    anlematrixre = winkelgesamt(matrixnew,Sensoren=["LUA","LLA"])
    anglestickright = anlematrixre1
    anglestickleft = anlematrixre
    plt.plot(anglestickleft)
    plt.plot(anglestickright)
    plt.show()

    rightpeak = []
    leftpeak = []
    for i in range(0,len(extractedstep)):
        posleft = (np.argmax(ACCleft[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        leftpeak.append(anglestickleft[(extractedstep[i,0])+posleft])
        posright = (np.argmax(ACCright[(extractedstep[i,0]):(extractedstep[i,1]),0]))
        rightpeak.append(anglestickright[(extractedstep[i,0])+posright])


    print rightpeak

    return rightpeak,leftpeak

#Timingprobleme sind aequivalent zum Passgang feature


        







