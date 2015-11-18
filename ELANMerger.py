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

#daten laden die benoetigt werden


def create(elan):
    elan = pd.concat([elan.iloc[:,[0,2,3,4,5]]])

    elan = elan.as_matrix()
    newdata = elan
    substract = elan[0,1]
    for i in range(0,len(elan)):
        newdata[i,1]= elan[i,1]-substract
        newdata[i,2]= elan[i,2]-substract
        newdata[i,4]=i
    return  elan