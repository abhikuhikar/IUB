# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 02:03:52 2018

@author: Abhilash
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment5\data\data\\"

twitter = io.loadmat(base_path + 'twitter.mat')
X_tr = twitter['Xtr']
X_te = twitter['Xte']
Y_tr = twitter['YtrMat']
Y_te = twitter['YteMat']

np.random.seed(7)
B = np.random.rand(891,50)
Theta = np.random.rand(50,773)

#learning weights from training data
for i in range(500):
    den = np.dot(B,Theta)
    den[den == 0] = 1e-3
    B = B*np.dot((X_tr/den),Theta.T)
    B = B/np.dot(np.ones([B.shape[0],B.shape[0]]),B)
    
    den = np.dot(B,Theta)
    den[den == 0] = 1e-3
    Theta = Theta*np.dot(B.T,(X_tr/den))
    Theta = Theta/np.dot(np.ones([Theta.shape[0],Theta.shape[0]]),Theta)

#learning weights from test data
Theta2 = np.random.rand(50,193)
for i in range(500):
    den = np.dot(B,Theta2)
    den[den == 0] = 1e-4
    Theta2 = Theta2*np.dot(B.T,(X_te/den))
    Theta2 = Theta2/np.dot(np.ones([Theta2.shape[0],Theta2.shape[0]]),Theta2)

#single perceptron training
alpha = 0.0065
W = np.random.uniform(0,5, (3,Theta.shape[0]))
b = np.random.uniform(0,5,(3,1))
error_plot = []

for i in range(1000):
    Z = np.dot(W, Theta) + b
    Y_hat = np.exp(Z)
    Y_hat = Y_hat/np.sum(Y_hat,axis=0).reshape(1,-1)
    error = -np.sum(Y_tr*np.log(Y_hat))
    #backpropagation
    delta = np.dot((Y_hat-Y_tr),Theta.T)
    delta_b = np.dot((Y_hat-Y_tr), np.ones([Y_tr.shape[1],1]))
    W = W - alpha*delta
    b = b - alpha*delta_b
    error_plot.append(error)

#prediction on test data
Z = np.dot(W, Theta2) + b
Y_hat_test = np.exp(Z)
Y_hat_test = Y_hat_test/np.sum(Y_hat_test,axis=0).reshape(1,-1)

train_accuracy = np.sum(np.argmax(Y_hat,axis=0) == np.argmax(Y_tr,axis=0))/Y_tr.shape[1]
test_accuracy = np.sum(np.argmax(Y_hat_test,axis=0) == np.argmax(Y_te,axis=0))/Y_te.shape[1]
plt.plot(error_plot)

