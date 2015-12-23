__author__ = 'Sebastian'
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
x1bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1126\Labels.csv', sep = "\t")
x1bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1126\dataMatrix.csv', sep = "\t")

x2bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1203\Labels.csv', sep = "\t")
x2bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1203\dataMatrix.csv', sep = "\t")


x3bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1211\Labels.csv', sep = "\t")
x3bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1211\dataMatrix.csv', sep = "\t")


x4bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1217\Labels.csv', sep = "\t")
x4bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\1217\dataMatrix.csv', sep = "\t")


x5bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\12172\Labels.csv', sep = "\t")
x5bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\Labels and data\ID004\12172\dataMatrix.csv', sep = "\t")

datax11 = pd.DataFrame(x1bus1.as_matrix())
datax12 = pd.DataFrame(x1bus2.as_matrix())
datax21 = pd.DataFrame(x2bus1.as_matrix())
datax22 = pd.DataFrame(x2bus2.as_matrix())
datax31 = pd.DataFrame(x3bus1.as_matrix())
datax32 = pd.DataFrame(x3bus2.as_matrix())
datax41 = pd.DataFrame(x4bus1.as_matrix())
datax42 = pd.DataFrame(x4bus2.as_matrix())
datax51 = pd.DataFrame(x5bus1.as_matrix())
datax52 = pd.DataFrame(x5bus2.as_matrix())


Labels = pd.concat([datax21,datax31,datax41,datax51])
Data =pd.concat([datax22,datax32,datax42,datax52])

print Data
print Labels
np.savetxt("dataAll.csv",Data,delimiter="\t")
np.savetxt("LabelsAll.csv",Labels,delimiter="\t",fmt="%s")