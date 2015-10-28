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
import pylab as plb
import Labeling
import Classifier
import PeakDetection
import scipy.io

# person: 0 = Sebastian, 1 = Tobias, 2 = Matthias
person = 0

if(person == 0):
    rawData = pd.read_csv(r'C:\Users\Sebastian\Desktop\VectorenID001.csv', sep='\t')
    Vectoren = rawData.as_matrix()
    rawData = pd.read_csv(r'C:\Users\Sebastian\Documents\PycharmMES\mespraktikum\merged.csv', sep='\t')
    Vectorenberechnet = rawData.as_matrix()
    rowData = pd.read_csv(r'C:\Users\Sebastian\Desktop\20150901\merged.csv', sep='\t')
    dataMatrix = rowData.as_matrix()
    label = scipy.io.loadmat(r'C:\Users\Sebastian\Desktop\Labels\MARKER_10.mat')
    label = label['seg']
elif (person == 1):
    print 'kann nix'
elif (person == 2):
    rawData = pd.read_csv('', sep='\t')
    Vectoren = rawData.as_matrix()
    rowData = pd.read_csv('/Users/MatthiasFuchs/Desktop/Daten+Labels/NWDaten/ID001/20150901/merged.csv', sep='\t')
    dataMatrix = rowData.as_matrix()
    label = scipy.io.loadmat(r'/Users/MatthiasFuchs/Desktop/Daten+Labels/Labels/ID001/MARKER_10.mat')
    label = label['seg']
else:
    print 'ungueltige Person'


matrixnew = Labeling.labeldata(dataMatrix,label)

Sensor1 = range(11,14)
Sensor2 = range(24,27)
Sensor3 = range(27,40)
Sensor4 = range(50,53)
Sensor5 = range(63,66)
Sensor6 = range(76,79)

Sensor = Sensor1+Sensor2+Sensor3+Sensor4+Sensor5+Sensor6
print Sensor
#trennen der Daten in Trainings und Testdaten fuer die Klassifizierer






print "classify start"
xlf, X_train, X_test, y_train, y_test = Classifier.classify(Vectoren,Sensor,classifier="Forest")
print xlf.score(X_test,y_test)
Classifier.printclassifier(xlf,X_train, X_test, y_train, y_test,Sensor)
Classifier.compareclassifier(Vectoren,Sensor)
print "classify finished"
#short= dataMatrix[46000:,:]
#np.savetxt('testdataproband001.csv', fmt=['%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f'] ,X= short, delimiter='\t')


