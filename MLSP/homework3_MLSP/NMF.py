# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:30:07 2018

@author: Abhilash
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment3\data\data\\"
source = base_path + "trs.wav"

trs_wave, sr1 = librosa.load(source, sr = None)
trn_wave, sr1 = librosa.load(base_path + "trn.wav", sr = None)
x_nmf_wave, sr1 = librosa.load(base_path + "x_nmf.wav", sr = None)

N = 1024


#Create a DFT matrix of NxN
def create_DFT_Matrix(N):
    DFT_matrix = np.array([[np.exp(-1j*(2*np.pi*f*n/N)) for f in range(0,N)] for n in range(0,N)])
    return DFT_matrix

def create_Inverse_DFT_Matrix(N):
    DFT_inv_matrix = np.array([[np.exp(1j*(2*np.pi*f*n/N)) for f in range(0,N)] for n in range(0,N)])/N
    return DFT_inv_matrix

#Using the code from homework 2
#Creating the data matrix by multiplying Hann window to size N and shifting the window each time by N/2
# X is the input matrix and N is the size of the Hann window
def create_DataMatrix_Using_Hann_Window(source, N):
    n = 0
    #Creating a Hann window
    Hann_window = np.hanning(N)
    X_DataMatrix = []
    while n+N <= len(source):
        x_window = source[n:n+N]
        x_vector = np.multiply(x_window, Hann_window)
        X_DataMatrix.append(x_vector)
        n = n + int(N/2)
    X_DataMatrix = np.array(X_DataMatrix)
    return X_DataMatrix.T

def NMF_learn(X, k):
    W = np.random.rand(X.shape[0], k) #F x k
    H = np.random.rand(k, X.shape[1]) # k x T
    Ones = np.ones(X.shape) # F x T
    for i in range(0, 2000):
        W = np.multiply(W, np.multiply(np.dot(X/(np.dot(W,H)), H.T), (1/(np.dot(Ones, H.T)))))
        H = H * (np.dot(W.T, (X/np.dot(W, H))))/(np.dot(W.T, Ones))
        
    return W, H

def NMF_learn_onlyH(X, k, W):
    H = np.random.rand(k, X.shape[1]) # k x T
    Ones = np.ones(X.shape) # F x T
    for i in range(0, 2000):
        H = H * (np.dot(W.T, (X/np.dot(W, H))))/(np.dot(W.T, Ones))
        
    return H

def reverse_Overlap_and_Add(X, N):
    X_out = X[:(X.shape[0] - int(N/2)),0]
    for col in range(1,X.shape[1]):
        X_col = X[(X.shape[0] - int(N/2)):,col-1] + X[0:int(N/2),col]
        X_out = np.concatenate((X_out,X_col))
    
    X_out = X_out.reshape(-1)
    
    return X_out

#DFT matrix
DFT_matrix = create_DFT_Matrix(N)

trs_highDimensional = create_DataMatrix_Using_Hann_Window(trs_wave, N)

trs_spectogram = np.dot(DFT_matrix, trs_highDimensional)
trs_spectogram_half = trs_spectogram[0:513,:]
S = np.abs(trs_spectogram_half)

W_S, H_S = NMF_learn(S, 30)


#Repeat the procedure for the trn.wav
trn_highDimensional = create_DataMatrix_Using_Hann_Window(trn_wave, N)

trn_spectogram = np.dot(DFT_matrix, trn_highDimensional)
trn_spectogram_half = trn_spectogram[0:513,:]
Noise = np.abs(trn_spectogram_half)

W_N, H_N = NMF_learn(Noise, 30)

#Separate sources in x_nmf.wav using above two basis vectors
X_highDimensional = create_DataMatrix_Using_Hann_Window(x_nmf_wave, N)
X_spectogram = np.dot(DFT_matrix, X_highDimensional)
X_spectogram_half = X_spectogram[0:513,:]
Y_spectogram = np.abs(X_spectogram_half)

W = np.concatenate((W_S, W_N), axis = 1)
H = NMF_learn_onlyH(Y_spectogram, 60, W)

Magnitude_mask = np.dot(W_S, H[0:30, :])/np.dot(W, H)

Speech_estimate = Magnitude_mask * X_spectogram_half
Speech_conjugate = np.flip(np.conjugate(Speech_estimate[1:512, :]), axis = 0)

Speech = np.concatenate((Speech_estimate, Speech_conjugate), axis = 0)

Speech_recovered = np.dot(create_Inverse_DFT_Matrix(N), Speech)

Speech_time_domain = reverse_Overlap_and_Add(Speech_recovered.real, N)

output_sound = base_path + 'recoveredSpeech.wav'
librosa.output.write_wav(output_sound,Speech_time_domain, sr = sr1)

try_DFT = np.random.rand(16,1)
try_spectogram = np.dot(create_DFT_Matrix(16), try_DFT)

#Plot start

#Ploting original signal
x_range = range(trs_wave.shape[0])
plt.figure(figsize= (6,3))
plt.title("Original Source signal")
plt.plot(x_range, trs_wave)
plt.show()

#Ploting noise signal
x_range = range(trn_wave.shape[0])
plt.figure(figsize= (6,3))
plt.title("Original Noise signal")
plt.plot(x_range, trn_wave)
plt.show()

#Ploting test signal
x_range = range(x_nmf_wave.shape[0])
plt.figure(figsize= (6,3))
plt.title("Test signal")
plt.plot(x_range, x_nmf_wave)
plt.show()

#Ploting recovered signal
x_range = range(Speech_time_domain.shape[0])
plt.figure(figsize= (6,3))
plt.title("Recovered Speech signal")
plt.plot(x_range, Speech_time_domain)
plt.show()
#Plot end