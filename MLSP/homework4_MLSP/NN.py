# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:18:21 2018

@author: Abhilash
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment4\data\data\\"

concentric = scipy.io.loadmat(base_path + 'concentric.mat')
concentricX = concentric['X']
concentricX = concentricX/np.sum(concentricX, axis=1).reshape(-1,1)
#logistic function g
def g(x):
    return 1/(1 + np.exp(-x))

#labels for the training data
Y = np.array([int(i > 50) for i in range(152)])

#adding the bias 1 to the input so that the input for the first layer is 3x152
X1 = np.concatenate((concentricX, np.ones([1,concentricX.shape[1]])))

#Feed forward network
#first layer

#learning parameters for the first layer
#W1 consists of W1, b1 where b1 is the bias vector for the first layer
np.random.seed(0)
W1 = np.random.randn(3,3)
#W1[:,2] = 0
W2 = np.random.randn(1,4) #weights with bias b2
#W2[:,3] = 0
alpha = 0.05
Y_hat = []
error_plot = []

for i in range(6000):
    #hidden layer
    Z1 = np.dot(W1, X1) #3x152 - Z1 is input to the hidden layer
    X2 = g(Z1)
#    X2 = X2 >= 0.5
    X2 = np.concatenate((X2, np.ones([1,X2.shape[1]])))

    #final layer
    Z2  = np.dot(W2, X2) #1x152
    Y_hat = g(Z2) #output of the network 1x152
    #backpropagation
    #final layer
    bp_error2 = (Y_hat - Y)*(Y_hat*(1 - Y_hat)) # (yhat - y)*g_dash(Z2)
    delta2 = np.dot(bp_error2,X2.T)
    W2 = W2 - alpha*delta2
    #hidden layer
    bp_error1 = np.dot(W2.T,bp_error2)*(X2*(1-X2))
    #We will ignore the last row as teh lsat row of delta1 corresponds to the bias b2
    delta1 = np.dot(bp_error1, X1.T)[0:3,:]
    W1 = W1 - alpha*delta1
    error = 0.5*np.dot((Y_hat-Y),(Y_hat-Y).T)
    error_plot.append(error)

print(W1)
print(W2)
print(error)

error_plot = np.array(error_plot).reshape(6000,1)
#error plot
plt.title('Error plot')
plt.plot(error_plot)
plt.show()