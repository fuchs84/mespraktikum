__author__ = 'Sebastian'
from scipy.signal import argrelmin,argrelmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn import svm
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import math
import FourierTransformation
import StepExtraction
import pylab as plb
import Init
import FeatureSelection
import Labeling
import Classifier
import PeakDetection
import scipy.io
import time as t
import FeatureKonstruktion
import ELANMerger




#Datum und Zeit
lt = t.localtime()
date = 'Datum: %2i.%2i.%4i' % (lt[2], lt[1], lt[0])
time = 'Zeit: %2i:%2i:%2i' % (lt[3], lt[4], lt[5])

#Personenauswahl
#person: 0 = Sebastian, 1 = Tobias, 2 = Matthias
person = 0
pathData = ''
pathLabel = ''
if(person == 0):
    print 'Sebastian'
    elan = pd.read_csv(r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID002\20151126\20151126ID002\20151126ID002.csv', sep = "\t")
    elan = pd.DataFrame(elan.as_matrix())
    elan = ELANMerger.create(elan)
    pathData = r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID002\20151126\merged.csv'
    pathLabel = r'C:\Users\Sebastian\Desktop\Labels\MARKER_10.mat'
    rawData = pd.read_csv(pathData, sep='\t')
    dataMatrix = rawData.as_matrix()
    label = scipy.io.loadmat(pathLabel)
    label = label['seg']
elif (person == 1):
    print 'Tobias'
elif (person == 2):
    print 'Matthias'
    pathData = r'/Users/MatthiasFuchs/Desktop/Daten+Labels/NWDaten/ID001/20150901/merged+vectors.csv'
    pathLabel = r'/Users/MatthiasFuchs/Desktop/Daten+Labels/Labels/ID001/MARKER_10.mat'
    rawData = pd.read_csv(pathData, sep='\t')
    dataMatrix = rawData.as_matrix()
    label = scipy.io.loadmat(pathLabel)
    label = label['seg']
else:
    print 'ungueltige Person'

comment = ''
#History
fd = open('History.txt','a')
history = date + ' ' + time + '\n'
fd.write(history)
history = 'Kommentar: '+ comment + '\n'
fd.write(history)
history = 'Datenpfad: ' + pathData + '\n'
fd.write(history)
history = 'Labelpfad: ' + pathLabel + '\n'
fd.write(history)
fd.close()
#Daten + Label
#matrixnew = Labeling.labeldata(dataMatrix,label)
#matrixnew = Labeling.selectLabel(matrixnew[0:30000,:],label,[2,8])
#steparray, x = StepExtraction.stepDetectionback(matrixnew)
#passg = Init.getData(matrixnew,sensors=["RNS","LUF"],datas=["acc"],specifiedDatas=["z"])
#Datenauswahl
print(elan)
Matrix= dataMatrix
print Matrix[1854:,0]
newvalue = (dataMatrix[:,0])
timestamp =  []
duration = []

FeatureKonstruktion.aufrechtgehen(dataMatrix)
#ELANMerger.MatrixforAndroid(dataMatrix[1854:,:],elan)


print "steps"
print timestamp
print duration





#np.savetxt('selectedDatapcaVec05000.csv', pca[0:5000, :], delimiter='\t')
#np.savetxt('selectedDatamag.csv', matrixnew[0:20000, :], delimiter='\t')
plt.subplot(4,1,1)
plt.title("passgang")
#plt.plot(passgang)

plt.subplot(4,1,2)
plt.title("Stockaufsatz")
#plt.plot(rightpeak,label="Rechter Stock")
#plt.plot(leftpeak,label ="Linker Stock")
plt.subplot(4,1,3)
plt.title("Schulter")
#plt.plot(a,label= "Linkeschulter")
#plt.plot(b, label= "Rechteschulter")
plt.subplot(4,1,4)
plt.title("Arm")
#plt.plot(c,label= "Linker Arm")
#plt.plot(d, label= "Rechter Arm")




plt.show()
#Trennen der Daten in Trainings und Testdaten fuer die Klassifizierer
#clf,X_train, X_test, y_train, y_test = Classifier.classify(matrixnew,Sensor,classifier="AdaBoost")

#Starten des Klassifizierers
print "classify start"
#Classifier.printclassifier(xlf, matrixnew[:,Sensor], matrixnew[:,1], matrixnew[:,1],Sensor)
#Classifier.compareclassifier(matrixnew,Sensor)
print "classify finished"
#short= dataMatrix[46000:,:]
np.savetxt('testdataproband001.csv', fmt=['%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f'] ,X= short, delimiter='\t')

fd = open('History.txt','a')
fd.write('\n \n')
fd.close()
