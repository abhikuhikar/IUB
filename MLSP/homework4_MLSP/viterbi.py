# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:40:59 2018

@author: Abhilash
"""

import librosa
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment4\data\data\\"

piano_clap, sr1 = librosa.load(base_path + "Piano_Clap.wav", sr = None)

mfcc_mat = scipy.io.loadmat(base_path + 'mfcc.mat')
mfccX = mfcc_mat['X']

musigma = scipy.io.loadmat(base_path + 'MuSigma.mat')
mus = musigma['mX']
mu_piano = mus[:,0]
mu_claps = mus[:,1]
sigmas = musigma['Sigma']
cov_piano = sigmas[:,:,0]
cov_claps = sigmas[:,:,1]

def MultivariateGaussian(x, mu, cov):
    pdf = []
    for i in range(x.shape[1]):
        x_vector = x[:,i].reshape(1,-1)
        n_constant = (1/np.power(2*np.pi, (cov.shape[0])/2))*np.power(np.linalg.det(cov), -0.5)
        pdf.append(n_constant*np.exp((-0.5)*np.dot(np.dot((x_vector-mu), np.linalg.inv(cov)), (x_vector-mu).T)))
    return np.array(pdf).reshape(-1,)

p_piano = MultivariateGaussian(mfccX, mu_piano.reshape(1,-1), cov_piano)
p_claps = MultivariateGaussian(mfccX, mu_claps.reshape(1,-1), cov_claps)

P = np.array([p_piano,p_claps])
P = P / P.sum(axis = 0).reshape(1,-1)


#Detection result
plt.figure(figsize= (5,4))
plt.title('Detection result')
plt.imshow(P, interpolation='none', aspect ='auto')
plt.show()

#transition matrix
T = np.array([[0.9,0.1],[0,1]])


P_bar = np.array(P)
for i in range(P.shape[1]-1):
    b = np.argmax(P_bar[:,i])
    P_bar[:,i+1] = T[b,:]*P[:,i+1] 
    P_bar[:,i+1] = P_bar[:,i+1]/np.sum(P_bar[:,i+1])

#Detection result
plt.figure(figsize= (5,4))
plt.title('Smoothened Detection result')
plt.imshow(P_bar, interpolation='none', aspect ='auto')
plt.show()

P_bar_viterbi = np.array(P)
#viterbi
B = np.zeros(P.shape)
for i in range(P.shape[1]-1):
    b0 = np.argmax(T[:,0]*P_bar_viterbi[:,i])
    b1 = np.argmax(T[:,1]*P_bar_viterbi[:,i])
    B[0,i+1] = b0
    B[1,i+1] = b1
    P_bar_viterbi[0,i+1] = T[b0,0]*P_bar_viterbi[b0,i]*P[0,i+1]
    P_bar_viterbi[1,i+1] = T[b1,1]*P_bar_viterbi[b1,i]*P[1,i+1]
    P_bar_viterbi[:,i+1] = P_bar_viterbi[:,i+1]/np.sum(P_bar_viterbi[:,i+1])
    
#backtrack
for i in range(B.shape[1]-1):
    b = np.argmax(P_bar_viterbi[:, 961-i])
    b_prev = int(B[b,961-i])
    P_bar_viterbi[b_prev, 961-i-1] = 1
    P_bar_viterbi[1-b_prev, 961-i-1] = 0
    
#Detection result
plt.figure(figsize= (5,4))
plt.title('viterbi Clap Probability result')
#plt.imshow(P_bar_viterbi, interpolation='none', aspect ='auto')
plt.plot(P_bar_viterbi[1,:])
plt.show()

#Detection result
plt.figure(figsize= (5,4))
plt.title('Viterbi Detection result')
plt.imshow(P_bar_viterbi, interpolation='none', aspect ='auto')
plt.show()
