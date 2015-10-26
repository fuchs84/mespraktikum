__author__ = 'MatthiasFuchs'
import numpy as np
import matplotlib.pyplot as plt
import pywt as pw
import pandas as pd
import scipy as cp
from scipy import signal


#Testsignal
N = 512 # Sample count
fs = 128 # Sampling rate
st = 1.0 / fs # Sample time
t = np.arange(N) * st # Time vector

signal1 = \
1   *np.cos(2*np.pi * t) *\
2   *np.cos(2*np.pi * 4*t) *\
0.5 *np.cos(2*np.pi * 0.5*t)*\
2   *np.cos(2*np.pi * 20*t)

signal2 = \
0.25*np.sin(2*np.pi * 2.5*t) +\
0.25*np.sin(2*np.pi * 3.5*t) +\
0.25*np.sin(2*np.pi * 4.5*t) +\
0.25*np.sin(2*np.pi * 5.5*t)
#End Testsignal

rowData = pd.read_csv('/Users/MatthiasFuchs/Desktop/Daten+Labels/Testdaten/walk001.csv', sep= '\t')
data = rowData.as_matrix()


def computeWT(signal, wavelet = 'haar', level = 3):
    coefs = pw.wavedec(signal , wavelet, level=level, mode='zpd')
    return coefs

def computeIWT(coefs, wavelet = 'haar'):
    signal = pw.waverec(coefs, wavelet, mode= 'zpd')
    return signal

def computeMedianWT(coefs, level, median = 1):
    newCoefs = coefs[:]
    if(1 <= level <= len(coefs)):
        newCoefs[level] = cp.signal.medfilt(newCoefs[level], median)
    else:
        print 'level ungueltig'
    return newCoefs

def plotWTData(signal, coefs, newSignal, newCoefs):
    level = len(coefs)
    newLevel = len(newCoefs)
    plt.subplot(4, 1, 1)
    plt.plot(signal)
    plt.subplot(4, 1, 2)
    for i in range(1, level):
        plt.plot(coefs[i])
    plt.subplot(4, 1, 3)
    plt.plot(newSignal)
    plt.subplot(4, 1, 4)
    for i in range(1, newLevel):
        plt.plot(newCoefs[i])
    plt.show()

signal = data[:, 2]
coefs = computeWT(signal, 'db8', 4)
newCoefs = computeMedianWT(coefs, 2, 5)
newCoefs = computeMedianWT(newCoefs, 3, 5)
newCoefs = computeMedianWT(newCoefs, 4, 5)

newSignal = computeIWT(newCoefs, 'db8')
plotWTData(signal, coefs, newSignal, newCoefs)

# signal = data[:, 15]
#
# coefs = pw.wavedec(signal , 'db8', level=3, mode='per')
#
# processed = coefs[:]
#
# processed[1] = cp.signal.medfilt(processed[1], 31)
# processed[2] = cp.signal.medfilt(processed[2], 31)
# processed[3] = cp.signal.medfilt(processed[3], 31)
#
# signal_processed = pw.waverec(processed, 'db8', mode='per')
#
# plt.subplot(4,1,1)
# plt.plot(signal)
# plt.subplot(4,1,2)
# plt.plot(coefs[1], color='green')
# plt.plot(coefs[2], color='blue')
# plt.plot(coefs[3], color='red')
# plt.subplot(4,1,3)
# plt.plot(processed[1], color='green')
# plt.plot(processed[2], color='blue')
# plt.plot(processed[3], color='red')
# plt.subplot(4,1,4)
# plt.plot(signal_processed)
# plt.show()