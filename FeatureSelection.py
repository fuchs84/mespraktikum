__author__ = 'MatthiasFuchs'

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
import Classifier
import numpy as np
from sklearn.decomposition import PCA


def featureSelectionSelectKBest(data, Featurenumber):
    label = data[:,1]
    datanew = data[:,2:]
    for i in range(0,len(datanew)):
        datanew[i] = map(abs, datanew[i])
    size = Featurenumber
    selector = SelectKBest(chi2, k=size).fit(data[:,2:],data[:,1])
    print selector.get_support(True)
    X_new = selector.fit_transform(datanew, label)
    data[:,2:size+2] = X_new
    return data[:,:size+2]

# Sucht die besten Features und gibt die groesse des Datansaetz zurueck
def featureSelectionTree(data):
    label = data[:,1]
    datanew = data[:,2:]
    for i in range(0,len(datanew)):
        datanew[i] = map(abs, datanew[i])

    clf = ExtraTreesClassifier()
    X_new = clf.fit(datanew, label).transform(datanew)
    size = len(X_new[0])
    data[:,2:size+2] = X_new
    return data[:,:size+2], size

def featureSelectionVarianceThreshold(data, probability = 0.8):
    dataRaw = data[:, 2:]
    sel = VarianceThreshold(threshold=(probability*(1 - probability)))
    dataNew = sel.fit_transform(dataRaw)
    return np.c_[data[:, :2], dataNew]


def featureSelectionPCA(data, components):
    dataRaw = data[:, 2:]
    label = data[:, 1]
    sel = PCA(n_components= components, copy=True)
    dataNew = sel.fit_transform(dataRaw, label)
    return  np.c_[data[:, :2], dataNew]


#Gibt die wichtigsten Features eines gewaehlten Klassifzierers zurueck
def getNbestTreeFeaturesPos(data, n,  Klassifizierer="Forest"):
    end = len(data[1,:])
    xlf, X_train, X_test, y_train, y_test = Classifier.classify(data,range(2,end),classifier=Klassifizierer)
    z = (xlf.feature_importances_)
    z= np.array(z)
    k = z.argsort()[-n:][::-1]
    newlist = [x+2 for x in k]
    return newlist
