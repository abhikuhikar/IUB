# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 00:24:37 2018

@author: Abhilash
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment3\data\data\\"

eeg = scipy.io.loadmat(base_path + 'eeg.mat')

x_train = eeg['x_train']
y_train = eeg['y_train']
x_test = eeg['x_te']
y_test = eeg['y_te']

#hyperparameters of the algoeithm
L = 200
k = 80
M = 225
#Create a DFT matrix of NxN
def create_DFT_Matrix(N):
    DFT_matrix = np.array([[np.exp(-1j*(2*np.pi*f*n/N)) for f in range(0,N)] for n in range(0,N)])
    return DFT_matrix

def create_Inverse_DFT_Matrix(N):
    DFT_inv_matrix = np.array([[np.exp(1j*(2*np.pi*f*n/N)) for f in range(0,N)] for n in range(0,N)])/N
    return DFT_inv_matrix

#create data matrix using a blackman window
def create_DataMatrix_Using_Blackman_Window(source, N, hop):
    n = 0
    #Creating a Hann window
    Hann_window = np.blackman(N)
    X_DataMatrix = []
    while n+N <= len(source):
        x_window = source[n:n+N]
        x_vector = np.multiply(x_window, Hann_window)
        X_DataMatrix.append(x_vector)
        n = n + hop
    X_DataMatrix = np.array(X_DataMatrix)
    return X_DataMatrix.T

#create a DFT matrix of 64 x 64
DFT_Matrix = create_DFT_Matrix(64)

#create a STFT matrix for a given sample for all the three channels
def STFT_on_channels(sample):
    stft = []
    for i in range(0,3):
        windowed_channel = create_DataMatrix_Using_Blackman_Window(sample[:,i], 64, 48)
        channel_stft = np.abs(np.dot(DFT_Matrix, windowed_channel))[0:33, :]
        stft.append(channel_stft)
    return np.array(stft)

#returns the vectorized STFT mu wave (i.e only 3rd to 7th row in STFT matrix) for all the channels of the given data sample 
def vectorizedSTFTMuWave(sample):
    #create a STFT matrix for a given sample for all the three channels
    stft_mu_vector = []
    for i in range(0,3):
        windowed_channel = create_DataMatrix_Using_Blackman_Window(sample[:,i], 64, 48)
        #selecting only mu waves from the STFT matrix and vectorizing it for that channel
        channel_stft_mu = np.abs(np.dot(DFT_Matrix, windowed_channel))[2:7, :].reshape(-1,1) 
        stft_mu_vector.append(channel_stft_mu)
    return np.array(stft_mu_vector).reshape(-1,1)

#create a #test samples x #train samples matrix as the hamming matrix 
def HammingDistanceVector(test_data, train_data):
    hammingDist = np.array(np.random.rand(test_data.shape[1],train_data.shape[1]))
    for i in range(0, test_data.shape[1]):
        for j in range(0, train_data.shape[1]):
            hammingDist[i][j] = np.sum(test_data[:,i] ^ train_data[:,j]) / -2
    return hammingDist.astype(int)

#returns the labels for all the data samples
def kNN(hammingVector, k):
    labels = []
    for i in range(0, hammingVector.shape[0]):
        sum_ = np.sum(y_train[hammingVector[i,0:k]])
        if 2*k - sum_ > (k/2):
            labels.append(1)
        else:
            labels.append(2)
    
    return np.array(labels).reshape(-1,1)

def solve_kNN(k, L):    
    stft_mu_wave_vectored_data_train = []
    for i in range(0, x_train.shape[2]):
        stft_mu_wave_vectored_data_train.append(vectorizedSTFTMuWave(x_train[:,:,i]))
    
    stft_mu_wave_vectored_data_train = np.array(stft_mu_wave_vectored_data_train).T.reshape(225,112)
    
    #initialize vector A rendomly with 100 x 225 shape
    A = np.random.seed(L)
    A = np.random.uniform(-2,2,(L, 225))
    #normalize all the row vectors such that their sum is 1
    A = A / np.sum(A, axis=1).reshape(-1,1)
    X_reduced_vector_train = np.sign(np.dot(A, stft_mu_wave_vectored_data_train)).astype(int)
    
    #for test data
    stft_mu_wave_vectored_data_test = []
    for i in range(0, x_test.shape[2]):
        stft_mu_wave_vectored_data_test.append(vectorizedSTFTMuWave(x_test[:,:,i]))
    
    stft_mu_wave_vectored_data_test = np.array(stft_mu_wave_vectored_data_test).T.reshape(225,28)
    
    X_reduced_vector_test = np.sign(np.dot(A, stft_mu_wave_vectored_data_test)).astype(int)
    hammingDist_vector = HammingDistanceVector(X_reduced_vector_test, X_reduced_vector_train)
    
    #find k nearest neighbours for all the test data samples - sorting and returning the indices of the sorted values
    hammingDist_vector = hammingDist_vector.argsort(axis = 1)
    
    knn = kNN(hammingDist_vector, k)
    
    return knn

knn = solve_kNN(k, L)
misclassified = np.sum(np.abs(knn - y_test))
accuracy = (28 - misclassified)/28

print (accuracy)

'''
accuracy_vect = []
for i in range(0, L):
    knn= solve_kNN(i, L)
    misclassified = np.sum(np.abs(knn - y_test))
    accuracy = (28 - misclassified)/28
    accuracy_vect.append(accuracy*100)

plt.plot(accuracy_vect)
'''
"""
windowed = create_DataMatrix_Using_Blackman_Window(first_sample, 64, 48)
stft = np.abs(np.dot(create_DFT_Matrix(64), windowed))
stft = np.flip(stft[0:33, :], axis = 0)
"""