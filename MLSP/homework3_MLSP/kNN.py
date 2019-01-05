# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 09:43:49 2018

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
k = 25

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


def reverse_Overlap_and_Add(X, N):
    X_out = X[:(X.shape[0] - int(N/2)),0]
    for col in range(1,X.shape[1]):
        X_col = X[(X.shape[0] - int(N/2)):,col-1] + X[0:int(N/2),col]
        X_out = np.concatenate((X_out,X_col))
    
    X_out = X_out.reshape(-1)
    
    return X_out

#k nearest neighbours
    #test data 513x129 | train data 513x987 | output 129 x987
def kNN(X_test, X_train, k):
    nn_matrix = np.zeros((X_test.shape[1], X_train.shape[1]))
    for i in range(0, X_test.shape[1]):
        for j in range(0, X_train.shape[1]):
            nn_matrix[i][j] = np.sqrt(np.sum(np.power((X_train[:,j] - X_test[:,i]), 2)))
    
    return nn_matrix.argsort()[:,0:k]

#predict IBM for test data
def predictIBM(nn, B):
    D = np.zeros((B.shape[0], nn.shape[0]))
    for i in range(0, nn.shape[0]):
        D[:,i] = np.median(B[:,nn[i,:]], axis=1)
    return D

#DFT matrix
DFT_matrix = create_DFT_Matrix(N)

trs_highDimensional = create_DataMatrix_Using_Hann_Window(trs_wave, N)

trs_spectogram = np.dot(DFT_matrix, trs_highDimensional)
trs_spectogram_half = trs_spectogram[0:513,:]
S = np.abs(trs_spectogram_half)

#Repeat the procedure for the trn.wav
trn_highDimensional = create_DataMatrix_Using_Hann_Window(trn_wave, N)

trn_spectogram = np.dot(DFT_matrix, trn_highDimensional)
trn_spectogram_half = trn_spectogram[0:513,:]
Noise = np.abs(trn_spectogram_half)

#G
G = S + Noise

#Binary Mask matrix s.t it is 1 if S >= Noise
IBM = (S >= Noise).astype(int)

#Separate sources in x_nmf.wav using above two basis vectors
X_highDimensional = create_DataMatrix_Using_Hann_Window(x_nmf_wave, N)
X_spectogram = np.dot(DFT_matrix, X_highDimensional)
X = X_spectogram[0:513,:]
Y = np.abs(X)

NN_matrix = kNN(Y, G, k)

D = predictIBM(NN_matrix, IBM)

S_hat_test = D * X

S_hat_conjugate = np.flip(np.conjugate(S_hat_test[1:512, :]), axis = 0)
S_hat_test = np.concatenate((S_hat_test, S_hat_conjugate), axis = 0)
S_hat = np.dot(create_Inverse_DFT_Matrix(N), S_hat_test)

S_hat_time = reverse_Overlap_and_Add(S_hat.real, N)

output_sound = base_path + 'recoveredSpeechkNN.wav'
librosa.output.write_wav(output_sound,S_hat_time, sr = sr1)

#Ploting recovered signal
x_range = range(S_hat_time.shape[0])
plt.figure(figsize= (6,3))
plt.title("Recovered Speech signal")
plt.plot(x_range, S_hat_time)
plt.show()
#Plot end