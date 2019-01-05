# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Parts answered in this questio:
#Draw the histogram of s.wav
#Load two signals as vectors
#Standardize both signals by subtracting their sample means and by divid-
    #ing by their sample standard deviation.
#Calculate the Kurtosis of all the waves
#Which one is less Gaussian like according to the Kurtosis value
#Draw histograms of the two signals
#Compare them with the histogram of s.wav
#Turn in histograms, Kurtosis values and the verbal explaination of your decision

import librosa
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment1\AudioFiles\\"

source = base_path + 's.wav'
x1_source = base_path + 'x1.wav'
x2_source = base_path + 'x2.wav'

s, sr1 = librosa.load(source, sr=None)
x1, srx1 = librosa.load(x1_source, sr=None)
x2, srx2 = librosa.load(x2_source, sr=None)

#Draw the histogram of s.wav

#Standardize s.wav
s = s - np.mean(s)
s = s * (1/np.std(s))

#Standardize x1.wav
x1 = x1 - np.mean(x1)
x1 = x1 * (1/np.std(x1))

#Standardize x2.wav
x2 = x2 - np.mean(x2)
x2 = x2 * (1/np.std(x2))
    
#Plotting histograms of all the sound signals    
plt.hist(s)
plt.hist(x1)
plt.hist(x2)

#Calculating the Kurtosis values of all the waves
s_Kurtosis = np.mean(s**4) - 3.0
x1_Kurtosis = np.mean(x1**4 - 3.0)
x2_Kurtosis = np.mean(x2**4 - 3.0)

plt.hist(s, density=True)
plt.title('Histogram of s.wav')
plt.hist(x1, density=True)
plt.title('Histogram of x1.wav')
plt.hist(x2, density=True)
plt.title('Histogram of x2.wav')
