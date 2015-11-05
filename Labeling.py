__author__ = 'Sebastian'
from scipy.signal import argrelmin,argrelmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pylab as plb


def labeldata(dataMatrix, Matlablabel):
    label = Matlablabel
    label = label[:,(0,1,3)]

    for i in range(0,len(label)):
        for j in range(label[i,0],label[i,1]+1):
            try:
                dataMatrix[j,1]=(label[i,2])
            except IndexError:
                p=1

    return dataMatrix[1:,:]

def selectLabel(dataMatrix, Matlablabel, labelNumbers):
    label = Matlablabel
    output = np.empty(shape=(0, len(dataMatrix[0, :])))
    for i in range (0, len(label)):
        if(label[i, 3] in labelNumbers):
            start = label[i,0]
            stop = label[i,1]+1
            output = np.r_[output, dataMatrix[start:stop, :]]
    return output
