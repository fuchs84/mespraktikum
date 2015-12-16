__author__ = 'Sebastian'
from __builtin__ import int
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import pandas as pd
import math
import pylab as plb
import pandas as pd
import computeAngles as ca
import  StepExtraction
import FeatureKonstruktion

#daten laden die benoetigt werden


def create(elan):

    #elan = pd.concat([elan.iloc[:,[0,2,3,4,5]]])
    elan = pd.DataFrame(elan.iloc[:,[0,2,3,4,5]])
    elan = elan.as_matrix()


    return  elan

def MatrixforAndroid(datamatrix, elan):
    modifiedmatrix ,steps = StepExtraction.videosteps(datamatrix,elan)
    print steps
    print elan
    e,f = FeatureKonstruktion.handverkrampft(modifiedmatrix,steps[:,[0,1]])
    g,h,k = FeatureKonstruktion.aufrechtgehen(modifiedmatrix,steps[:,[0,1]])
    i,j =FeatureKonstruktion.Fussabrollen(modifiedmatrix,steps[:,[0,1]])
    passgang = FeatureKonstruktion.Passgang(modifiedmatrix,steps[:,[0,1]])
    rightpeak,leftpeak = FeatureKonstruktion.Stockaufsatz(modifiedmatrix,steps[:,[0,1]])
    a,b = FeatureKonstruktion.schulterbewegung(modifiedmatrix,steps[:,[0,1]])
    c,d = FeatureKonstruktion.armstreckung(modifiedmatrix,steps[:,[0,1]])


    elan = pd.DataFrame(elan[:,1:],index=elan[:,0])
    elanpass = np.array(elan.ix['1Passgang'])[:,3]
    elanstock = np.array(elan.ix['1Stockeinsatz'])[:,3]
    elanarm = np.array(elan.ix['1Armarbeit'])[:,3]
    elanschritt = np.array(elan.ix['2Schrittlaenge'])[:,3]
    elanschub = np.array(elan.ix['2Schub'])[:,3]
    elanverkrampfung = np.array(elan.ix['2Verkrampfung'])[:,3]
    elanober = np.array(elan.ix['2Oberkoerper'])[:,3]
    elantiming = np.array(elan.ix['2Timing'])[:,3]
    elanshwingen = np.array(elan.ix['3Vorschwingen'])[:,3]
    elanblick = np.array(elan.ix['3Blick'])[:,3]

    print len(elanarm)
    new = np.c_[steps,elanpass]
    new = np.c_[new,elanstock]
    new = np.c_[new,elanarm]
    new = np.c_[new,elanschritt]
    new = np.c_[new,elanschub]
    new = np.c_[new,elanverkrampfung]
    #new = np.c_[new,elanober]
    #new = np.c_[new,elantiming]
    new = np.c_[new,elanshwingen]
    new = np.c_[new,elanblick]


    plt.subplot(4,1,1)
    plt.title("passgang")
    plt.plot(passgang)

    plt.subplot(4,1,2)
    plt.title("Stockaufsatz")
    plt.plot(e)
    plt.plot(f)
    plt.subplot(4,1,3)
    plt.title("schulter")
    plt.plot(g)
    plt.plot(h, label= "Rechteschulter")
    plt.subplot(4,1,4)
    plt.title("Arm")
    plt.plot(i)
    plt.plot(j)
    #plt.plot(d, label= "Rechter Arm")
    plt.show()
    file = []
    label= []
    FeatureFile= np.c_[steps,passgang,rightpeak,leftpeak,a,b,c,d,e,f,g,h,i,j,k]
    for i in range(0,len(new)):
        temp = new[i,4]
        if temp=="3":
            new[i,4:]="3"
    for i in range(0,len(new)):
        if new[i,4]=="2" or new[i,4]=="1":
            print new[i,4]
            file.append(FeatureFile[i,:])
            label.append(new[i,:])
    labelpass = []
    filepass = []
    for i in range(0,len(new)):
        if new[i,4]=="2" or new[i,4]=="1" or new[i,4]== "3":
            filepass.append(FeatureFile[i,:])
            labelpass.append(new[i,:5])


    print(label)
    print(file)

    np.savetxt("20151127ID005features.csv",file,delimiter="\t")
    np.savetxt("20151127ID005labels.csv",label,delimiter="\t",fmt="%s")
    np.savetxt("20151127ID005featurespass.csv",filepass,delimiter="\t")
    np.savetxt("20151127ID005labelspass.csv",labelpass,delimiter="\t",fmt="%s")
    print "finished"
    plt.subplot(4,1,1)
    plt.plot(label[:,4:])
    plt.subplot(4,1,2)
    plt.plot(file[:,4:])
    plt.subplot(4,1,3)
    plt.plot(labelpass[:,4:])
    plt.subplot(4,1,4)
    plt.plot(filepass[:,4:])
    plt.show()

def wholedataelan(dataMatrix,elan):
    matrix, step = StepExtraction.videosteps(dataMatrix[:,:],elan)
    print step
    elan = pd.DataFrame(elan[:,1:],index=elan[:,0])
    elanpass = np.array(elan.ix['1Passgang'])[:,3]
    elanstock = np.array(elan.ix['1Stockeinsatz'])[:,3]
    elanarm = np.array(elan.ix['1Armarbeit'])[:,3]
    elanschritt = np.array(elan.ix['2Schrittlaenge'])[:,3]
    elanschub = np.array(elan.ix['2Schub'])[:,3]
    elanverkrampfung = np.array(elan.ix['2Verkrampfung'])[:,3]
    elanober = np.array(elan.ix['2Oberkoerper'])[:,3]
    elantiming = np.array(elan.ix['2Timing'])[:,3]
    elanshwingen = np.array(elan.ix['3Vorschwingen'])[:,3]
    elanblick = np.array(elan.ix['3Blick'])[:,3]

    print len(elanarm)
    new = np.c_[step,elanpass]
    new = np.c_[new,elanstock]
    new = np.c_[new,elanarm]
    new = np.c_[new,elanschritt]
    new = np.c_[new,elanschub]
    new = np.c_[new,elanverkrampfung]
    #new = np.c_[new,elanober]
    #new = np.c_[new,elantiming]
    new = np.c_[new,elanshwingen]
    new = np.c_[new,elanblick]
    for i in range(0,len(new)):
        temp = new[i,4]
        if temp=="3":
            print "test"

            new[i,4:]="3"
    labeldata = []
    for i in range(0,len(new)):
            if new[i,4]=="3" or new[i,4]=="2" or new[i,4]=="1":
                print new[i,4]
                labeldata.append(new[i,:])
    labeldata = np.array(labeldata)
    matrixsteps = pd.DataFrame(matrix[labeldata[0,0]:labeldata[0,1],:])
    matrix = pd.DataFrame(matrix)
    for i in range(1,len(labeldata)):
        matrixsteps = pd.concat([matrixsteps,matrix.iloc[labeldata[i,0]:labeldata[i,1],:]])

    label = []
    for i in range(0,len(labeldata)):
        for k in range(labeldata[i,0],labeldata[i,1]):
            labels = np.array(labeldata[i,4:])
            #labels = map(int,labels)
            label.append(labels)





    matrixstepspass = np.array(matrixsteps)
    labelpass = np.array(label)
    print matrixstepspass
    print labelpass[:,4]
    print(len(matrixstepspass))
    print(len(labelpass))
    plt.subplot(2,1,1)
    plt.plot(matrixstepspass[:,2:5])
    plt.subplot(2,1,2)
    plt.plot(labelpass[:,0])
    plt.show()
    np.savetxt("dataMatrixpass.csv",matrixstepspass,delimiter="\t")
    np.savetxt("Labelspass.csv",labelpass[:,0],delimiter="\t",fmt="%s")
    label =[]
    for i in range(0,len(labeldata)):
        temp = labeldata[i,4]
        if temp != "3":
            label.append(labeldata[i,:])

    label = np.array(label)
    print label
    matrix=np.array(matrix)
    matrixsteps = pd.DataFrame(matrix[label[0,0]:label[0,1],:])
    matrix = pd.DataFrame(matrix)
    for i in range(1,len(label)):
        matrixsteps = pd.concat([matrixsteps,matrix.iloc[label[i,0]:label[i,1],:]])
    label = label[:,:]


    labelnew = []
    for i in range(0,len(label)):
        for k in range(label[i,0],label[i,1]):
            labels = np.array(label[i,4:])
            #labels = map(int,labels)
            labelnew.append(labels)

    print(matrixsteps)
    print(labelnew)
    print len(matrixsteps)
    print len(labelnew)
    plt.subplot(2,1,1)
    plt.plot(matrixsteps.iloc[:,2:5])
    plt.subplot(2,1,2)
    plt.plot(labelnew)
    plt.show()
    np.savetxt("dataMatrix.csv",matrixsteps,delimiter="\t")
    np.savetxt("Labels.csv",labelnew,delimiter="\t",fmt="%s")