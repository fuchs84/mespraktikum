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




#Datum und Zeit
lt = t.localtime()
date = 'Datum: %2i.%2i.%4i' % (lt[2], lt[1], lt[0])
time = 'Zeit: %2i:%2i:%2i' % (lt[3], lt[4], lt[5])

#Personenauswahl
#person: 0 = Sebastian, 1 = Tobias, 2 = Matthias
person = 2
pathData = ''
pathLabel = ''
if(person == 0):
    print 'Sebastian'
    pathData = r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID001\20150901\merged.csv'
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
matrixnew = Labeling.labeldata(dataMatrix,label)
matrixnew = Labeling.selectLabel(matrixnew,label, [8, 2])
#Datenauswahl
matrixnew = Init.getData(matrixnew,datas=['rE1'], specifiedDatas=['z'])


np.savetxt('selectedData.csv', matrixnew[7000:9000, :], delimiter='\t')


#Trennen der Daten in Trainings und Testdaten fuer die Klassifizierer
#clf,X_train, X_test, y_train, y_test = Classifier.classify(matrixnew,Sensor,classifier="AdaBoost")

#Starten des Klassifizierers
print "classify start"
#Classifier.printclassifier(xlf, matrixnew[:,Sensor], matrixnew[:,1], matrixnew[:,1],Sensor)
#Classifier.compareclassifier(matrixnew,Sensor)
print "classify finished"
#short= dataMatrix[46000:,:]
#np.savetxt('testdataproband001.csv', fmt=['%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f'] ,X= short, delimiter='\t')


fd = open('History.txt','a')
fd.write('\n \n')
fd.close()
