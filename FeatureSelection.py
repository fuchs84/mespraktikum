__author__ = 'MatthiasFuchs'

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA


def featureSelection(data, Featurenumber):
    label = data[:,1]
    datanew = data[:,2:]
    for i in range(0,len(datanew)):
        datanew[i] = map(abs, datanew[i])
    size = Featurenumber
    X_new = SelectKBest(chi2, k=size).fit_transform(datanew, label)
    data[:,2:size+2] = X_new
    return data[:,:size+2]

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

