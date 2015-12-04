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

    #elan = pd.concat([elan.iloc[:,[0,2,3,4,5]]])
    elan = pd.DataFrame(elan.iloc[:,[0,2,3,4,5]])
    elan = elan.as_matrix()


    return  elan