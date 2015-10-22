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

xbox1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\VectorenID001.csv', sep='\t')
Vectoren = xbox1.as_matrix()
xbox1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\20150901\merged.csv', sep='\t')
dataMatrix = xbox1.as_matrix()
label = scipy.io.loadmat(r'C:\Users\Sebastian\Desktop\Labels\MARKER_10.mat')
label = label['seg']



matrixnew = Labeling.labeldata(dataMatrix,label)

Sensor1 = range(24,28)
Sensor2 = range(11,15)
Sensor3 = range(37,41)
Sensor4 = range(50,54)
Sensor5 = range(63,67)
Sensor6 = range(76,80)

Sensor = Sensor1+Sensor2+Sensor3
#Sensor= Sensor4+Sensor5+Sensor6
print Sensor
#trennen der Daten in Trainings und Testdaten fuer die Klassifizierer






print "classify start"
start =10000
traindata = Vectoren[:30000,:]
testdata = Vectoren[]


clfbayes= Classifier.classify(traindata,testdata,Sensor,classifier="SVM")

Classifier.printclassifier(clfbayes,traindata,testdata,Sensor)
print "classify finished"
#short= dataMatrix[46000:,:]
#np.savetxt('testdataproband001.csv', fmt=['%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f'] ,X= short, delimiter='\t')


