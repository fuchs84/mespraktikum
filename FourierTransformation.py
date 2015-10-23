__author__ = 'MatthiasFuchs'

import matplotlib.pyplot as plt
import numpy as np
import cmath

#Testsignal
N = 512 # Sample count
fs = 128 # Sampling rate
st = 1.0 / fs # Sample time
t = np.arange(N) * st # Time vector

signal1 = \
1   *np.cos(2*np.pi * t) *\
2   *np.cos(2*np.pi * 4*t) *\
0.5 *np.cos(2*np.pi * 0.5*t)

signal2 = \
0.25*np.sin(2*np.pi * 2.5*t) +\
0.25*np.sin(2*np.pi * 3.5*t) +\
0.25*np.sin(2*np.pi * 4.5*t) +\
0.25*np.sin(2*np.pi * 5.5*t)
#End Testsignal

T = 1.0/50

def computeValues(FFT, mode):
    N = len(FFT)
    freq = np.fft.fftfreq(N, T)
    if (mode == 0):
        bins = freq
        abs = np.abs(FFT)
        real = np.real(FFT)
        imag = np.imag(FFT)
    elif (mode == 1):
        bins = freq[:N/2]
        abs = np.abs(FFT[:N/2])
        real = np.real(FFT[:N/2])
        imag = np.imag(FFT[:N/2])
    return bins, abs, real, imag

def computeFFT(signal):
    N = len(signal)
    FFT = [i / (N/2) for i in np.fft.fft(signal)]
    return FFT

def computeIFFT(FFT):
    N = len(FFT)
    signal = [i/(N/2) for i in np.fft.ifft(FFT)]
    return signal

def computeZeroMean(data):
    mean = sum(data)/len(data)
    zeroMean = data - mean
    return zeroMean

def computeTimeAxis(data):
    N = len(data)
    axis = np.arange(N)*T
    return axis

def globalMax(data):
    value = data[0]
    matrixPosition = 0
    for i in range(0, len(data)-1):
        if (value < data[i]):
            value = data[i]
            matrixPosition = i
    return matrixPosition

FFT = computeFFT(signal2)
bins, abs = computeValues(FFT, 1)

print len(FFT)
print len(signal2)

plt.subplot(2, 1, 1)
plt.plot(signal2)
plt.subplot(2, 1, 2)
plt.plot(bins, abs)
plt.show()


# # FFT + bins + normalization
# bins = np.fft.fftfreq(N, st)
#
#
# fft  = [i / (N/2) for i in np.fft.fft(signal1)]
# fft2 = [i / (N/2) for i in np.fft.fft(signal2)]
#
# print len(fft)
# print fft
