__author__ = 'Sebastian'
from scipy.signal import argrelmin,argrelmax
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm,datasets
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import math


#Auswahl der Klassifiezierer: Bayes, Gradient, SVM und DecisionTree stehen zur Verfuegung
def classify(training, test, Sensoren, classifier="Bayes"):
    #X sind die Ausgewaehlten Features
    X= training[:,Sensoren]
    #Y sind die zu den Features passenden Labels
    Y=training[:,1]

    #Auswahl des Klassifizierers
    if classifier is "Bayes":
        clf = GaussianNB()
    elif classifier is "Gradient":
        clf = SGDClassifier()
    elif classifier is "Linear":
        clf = linear_model.LinearRegression()
    elif classifier is "LDA":
        clf= LDA()
    elif classifier is "SVM":
        clf = svm.SVC()
    elif classifier is "DecisionTree":
        clf = tree.DecisionTreeClassifier()
    else:
        print "kein korrekter Klassifizierer gewawehlt,Naive Bayes wurde verwendet"
        clf = GaussianNB()
    #Trainieren des Klassifitierers
    clf.fit(X, Y)
    lista = clf.predict(test[:,Sensoren])-test[:,1]
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    print "Score: " + str(sum(b))
    return clf
    #Ausgeben des  Ergebnises zum Vergleich der wahren Labels mit den Klassifizieten


#Gibt die Ergebnisse des Klassifizierers in einem Graphen aus
def printclassifier(clf, training, test,Sensoren):
    plt.subplot(3, 1, 1)
    plt.plot(clf.predict(test[:,Sensoren]), label = "Zugeteilte Labelwerte")
    plt.subplot(3, 1, 2)
    plt.plot(test[:,1], label = "Wahre Werte")

    plt.subplot(3, 1, 3)
    plt.plot(training[:,1], label = "Trainingsdaten")
    plt.show()


def confusemat(clf, test,Sensoren):

    cm = confusion_matrix(test[:,1],clf.predict(test[:,Sensoren]))
    plt.figure()
    plot_confusion_matrix(cm)

    #Erstellen normalisierte Konfusionsmatrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalisierte Konfusionsmatrix')

    plt.show()

def plot_confusion_matrix(cm, title='Konfusionsmatrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    legend =['sync', 'walking',  'turn',  'idle', 'nordic walking']
    tick_marks = np.arange(len(legend))
    plt.xticks(tick_marks, legend, rotation=45)
    plt.yticks(tick_marks, legend)
    plt.tight_layout()
    plt.ylabel('wahres label')
    plt.xlabel('klassifiziertes label')


def createsymmetricsplit(data, start, end):
    Vectoren = data
    start = start
    end =end
    label = Vectoren[start:end,0:2]
    testdata = np.c_[label,Vectoren[start:end,2:41]]
    testdata = np.c_[testdata,Vectoren[start:end,93:132]]
    traindata = np.c_[label,(Vectoren[start:end,54:93])]
    traindata = np.c_[traindata,(Vectoren[start:end,145:184])]
    return traindata, testdata