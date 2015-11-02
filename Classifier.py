__author__ = 'Sebastian'
from scipy.signal import argrelmin,argrelmax
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm,datasets
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import cross_validation
import numpy as np
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import math


#Auswahl der Klassifiezierer: Bayes, Gradient, SVM und DecisionTree stehen zur Verfuegung
def classify(data, Sensoren, classifier="Bayes"):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,Sensoren], data[:,1], test_size=0.4, random_state=0)


    #Auswahl des Klassifizierers
    if classifier is "Bayes":
        clf = BernoulliNB()
        history = 'Klassifizierer: Naive Bayes' + '\n'
    elif classifier is "Gradient":
        clf = SGDClassifier()
        history = 'Klassifizierer: Gradient Decent' + '\n'
    elif classifier is "Linear":
        clf = linear_model.LinearRegression()
        history = 'Klassifizierer: Linear Regression' + '\n'
    elif classifier is "LDA":
        clf= LDA()
        history = 'Klassifizierer: LDA' + '\n'
    elif classifier is "AdaBoost":
        clf= AdaBoostClassifier(n_estimators=100)
        history = 'Klassifizierer: AdaBoost' + '\n'
    elif classifier is "Forest":
        clf= RandomForestClassifier(n_estimators=100)
        history = 'Klassifizierer: Forest' + '\n'
    elif classifier is "SVM":
        clf = svm.SVC()
        history = 'Klassifizierer: SVN' + '\n'
    elif classifier is "DecisionTree":
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        history = 'Klassifizierer: DecisionTree' + '\n'
    else:
        print "kein korrekter Klassifizierer gewawehlt,Naive Bayes wurde verwendet"
        history = 'Klassifizierer: Fehler' + '\n'
        clf = GaussianNB()
    #Trainieren des Klassifitierers
    clf.fit(X_train, y_train)
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print "Score: " + str(score)
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'

    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    return clf,X_train, X_test, y_train, y_test
    #Ausgeben des  Ergebnises zum Vergleich der wahren Labels mit den Klassifizieten

def compareclassifier(data, Sensoren):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,Sensoren], data[:,1], test_size=0.4, random_state=0)

    history = 'Vergleiche Klassifizierer: \n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    print "Naive Bayes"
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: Naive Bayes' + '\n'

    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    print "Gradient Decent"
    clf = SGDClassifier()
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: Gradient Decent' + '\n'

    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    print "Linear"
    clf = linear_model.LinearRegression()
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: Linear Regression' + '\n'

    score = clf.score(X_test, y_test)
    #confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    print "LDA"
    clf= LDA()
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: LDA' + '\n'

    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    print "AdaBoost"
    clf= AdaBoostClassifier()
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: AdaBoost' + '\n'

    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    print "Randomforest"
    clf= RandomForestClassifier(n_estimators=10)
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: Randomforest' + '\n'
    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    # print "Support Vector Machine(Linear)"
    # clf = svm.SVC()
    # clf.fit(X_train,y_train)
    # print "Score: " + str(clf.score(X_test, y_test))
    # lista = clf.predict(X_test)-y_test
    # lista = map(abs, lista)
    # b = [1 if i else 0 for i in lista]
    #
    # history = 'Klassifizier: Support Vector Machine (Linear)' + '\n'
    #
    # score = clf.score(X_test, y_test)
    # confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))
    #
    # print "Fehlerkennung: " + str(sum(b))
    # print "Score: " + str(score)
    # print confusionMatrix
    #
    # history = history + 'Score: ' + str(score) + '\n'
    # history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    # history = history + 'Confusionsmatrix: ' + '\n'
    # history = history + str(confusionMatrix) + '\n'
    # history = history + '--------------------------------------------------' + '\n'
    # fd = open('History.txt','a')
    # fd.write(history)
    # fd.close()

    print "DecisionTree"
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train,y_train)
    print "Score: " + str(clf.score(X_test, y_test))
    lista = clf.predict(X_test)-y_test
    lista = map(abs, lista)
    b = [1 if i else 0 for i in lista]

    history = 'Klassifizier: Decision Tree' + '\n'

    score = clf.score(X_test, y_test)
    confusionMatrix = confusion_matrix(y_test,clf.predict(X_test))

    print "Fehlerkennung: " + str(sum(b))
    print "Score: " + str(score)
    print confusionMatrix

    history = history + 'Score: ' + str(score) + '\n'
    history = history + 'Fehlerkennung: ' + str(sum(b)) + '\n'
    history = history + 'Confusionsmatrix: ' + '\n'
    history = history + str(confusionMatrix) + '\n'
    history = history + '--------------------------------------------------' + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()


#Gibt die Ergebnisse des Klassifizierers in einem Graphen aus
def printclassifier(clf, X_test, y_train, y_test):
    plt.subplot(3, 1, 1)
    plt.plot(clf.predict(X_test), label = "Zugeteilte Labelwerte")
    plt.subplot(3, 1, 2)
    plt.plot(y_test, label = "Wahre Werte")

    plt.subplot(3, 1, 3)
    plt.plot(y_train, label = "Trainingsdaten")
    plt.show()

#erstellt die Konfusionsmatrix und normalisierte Konfusionsmatrix
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


#gibt Daten geteilt in Rechte und Linke haelfte zurueck
def createsymmetricsplit(data, start, end):
    Vectoren = data
    start = start
    end =end
    label = Vectoren[start:end,0:2]
    links = np.c_[label,Vectoren[start:end,15:54]]
    links = np.c_[links,Vectoren[start:end,106:145]]
    rechts = np.c_[label,(Vectoren[start:end,54:93])]
    rechts = np.c_[rechts,(Vectoren[start:end,145:184])]
    return links, rechts