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
    passgang = FeatureKonstruktion.Passgang(modifiedmatrix,steps[:,[0,1]])
    rightpeak,leftpeak = FeatureKonstruktion.Stockaufsatz(modifiedmatrix,steps[:,[0,1]])
    a,b = FeatureKonstruktion.schulterbewegung(modifiedmatrix,steps[:,[0,1]])
    c,d = FeatureKonstruktion.armstreckung(modifiedmatrix,steps[:,[0,1]])
    print len(steps)
    print len(passgang)
    print len(rightpeak)
    print len(leftpeak)
    print len(a)
    print len(c)
    print "Featurfile"
    print steps
    print passgang
    FeatureFile= np.c_[steps,passgang,rightpeak,leftpeak,a,b,c,d]
    print(FeatureFile)
    np.savetxt("20151126ID002features.csv",FeatureFile,delimiter="\t")