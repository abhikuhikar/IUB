# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:03:58 2018

@author: Abhilash
"""

import librosa
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment5\data\data\\"

s, sr1 = librosa.load(base_path + "trs.wav", sr = None)
n, sr1 = librosa.load(base_path + "trn.wav", sr = None)

F = 1024 #Frame size

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

#logistic function g
def g(x):
    return 1/(1 + np.exp(-x))

#signal with noise
x = s + n

#DFT matrix
DFT_matrix = create_DFT_Matrix(F)

#create STFT matrix
s_higherdimensional = create_DataMatrix_Using_Hann_Window(s, F)
s_STFT = np.dot(DFT_matrix, s_higherdimensional)
s_spectogram_half = s_STFT[0:513,:]
S = np.abs(s_spectogram_half)

#create STFT matrix
n_higherdimensional = create_DataMatrix_Using_Hann_Window(n, F)
n_STFT = np.dot(DFT_matrix, n_higherdimensional)
n_spectogram_half = n_STFT[0:513,:]
N = np.abs(n_spectogram_half)

#create STFT matrix
x_higherdimensional = create_DataMatrix_Using_Hann_Window(x, F)
x_STFT = np.dot(DFT_matrix, x_higherdimensional)
x_spectogram_half = x_STFT[0:513,:]
X = np.abs(x_spectogram_half)

#Ideal binary mask
M = S > N

HiddenUnits = 80
#Neural Network with input as X i.e. 513x786 matrix. Each input sample is 513 dimensional
np.random.seed(42)
#Weights for hidden layer
W1 = np.random.randn(HiddenUnits,513)
#biases for hidden layer
b1 = np.random.randn(HiddenUnits,1)
#Weights for output layer
W2 = np.random.randn(513,HiddenUnits) #weights with bias b2
#biases for output layer
b2 = np.random.randn(513,1)

#learning rate
alpha = 0.0005
M_hat = []
error_plot = []
epochs = 3000
for i in range(epochs):
    #hidden layer
    Z1 = np.dot(W1, X) + b1
    X2 = np.tanh(Z1)

    #final layer
    Z2  = np.dot(W2, X2) + b2
    M_hat = g(Z2) #output of the network 513x786

    #backpropagation
    #final layer
    bp_error2 = (M_hat - M)*(M_hat*(1 - M_hat)) # (yhat - y)*g_dash(Z2)
    delta2 = np.dot(bp_error2,X2.T)
    deltab2 = np.dot(bp_error2,np.ones((bp_error2.shape[1],1)))
    #update weights for output layer
    W2 = W2 - alpha*delta2
    #update bias for output layer
    b2 = b2 - alpha*deltab2
    #hidden layer
    bp_error1 = np.dot(W2.T,bp_error2)*(1-X2**2)
    delta1 = np.dot(bp_error1, X.T)
    deltab1 = np.dot(bp_error1, np.ones((bp_error1.shape[1],1)))
    W1 = W1 - alpha*delta1
    b1 = b1 - alpha*deltab1

    error = 0.5*sum(sum((M_hat-M)**2))
    error_plot.append(error)

test, sr1 = librosa.load(base_path + "tex.wav", sr = None)
clean, sr1 = librosa.load(base_path + "tes.wav", sr = None)

#create STFT matrix
test_higherdimensional = create_DataMatrix_Using_Hann_Window(test, F)
test_STFT = np.dot(DFT_matrix, test_higherdimensional)
test_spectogram_half = test_STFT[0:513,:]
X_test = np.abs(test_spectogram_half)

#create STFT matrix
clean_higherdimensional = create_DataMatrix_Using_Hann_Window(clean, F)
clean_STFT = np.dot(DFT_matrix, clean_higherdimensional)
clean_spectogram_half = clean_STFT[0:513,:]
X_clean = np.abs(clean_spectogram_half)

#estimate labels using feedforward
#hidden layer
Z1 = np.dot(W1, X_test) + b1
X2 = np.tanh(Z1)
#final layer
Z2  = np.dot(W2, X2) + b2
M_hat_test = g(Z2) #output of the network 513x786

#recover complex valued spectogram
X_testRecov = test_spectogram_half*M_hat_test
X_testRecov_conjugate = np.flip(np.conjugate(X_testRecov[1:512, :]), axis = 0)

Speech = np.concatenate((X_testRecov, X_testRecov_conjugate), axis = 0)
Speech_recovered = np.dot(create_Inverse_DFT_Matrix(F), Speech)
Speech_time_domain = reverse_Overlap_and_Add(Speech_recovered.real, F)

output_sound = base_path + 'recoveredSpeech.wav'
librosa.output.write_wav(output_sound,Speech_time_domain, sr = sr1)

#Calculating snr signal to noise ratio
clean = clean[0:Speech_time_domain.shape[0]]
num = np.dot(clean.T, clean)
den = np.dot((clean - Speech_time_domain).T,(clean - Speech_time_domain))
SNR = 10*np.log10(num/den)
#error plot
plt.title('Error plot')
plt.plot(error_plot)
plt.show()