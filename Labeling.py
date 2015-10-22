__author__ = 'Sebastian'
from scipy.signal import argrelmin,argrelmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pylab as plb


def labeldata(dataMatrix, label):
    label = label[:,(0,1,3)]

    for i in range(0,len(label)):
        for j in range(label[i,0],label[i,1]+1):
            try:
                dataMatrix[j,1]=(label[i,2])
            except IndexError:
                p=1

    return dataMatrix

