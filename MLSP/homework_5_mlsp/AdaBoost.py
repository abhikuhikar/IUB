# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:26:58 2018

@author: Abhilash
"""
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment5\data\data\\"

trX = io.loadmat(base_path + 'trX.mat')['trX']
trY = io.loadmat(base_path + 'trY.mat')['trY']

#some initializations
f_X = 0
M = 500
W_examples = np.ones((1,trX.shape[1]))
W_examples_total = []
beta_array = []
error_plot = []
classifier_outputs = []
perceptron_weights = []
perceptron_biases = []
for m in range(M):
    #single perceptron training
    alpha = 0.0005
    np.random.seed()
    W = np.random.uniform(-2,2, (1,2))
    b = np.random.uniform()
    
    for i in range(600):
        Z = np.dot(W, trX) + b
        Y_hat = np.tanh(Z)
        error = np.dot(W_examples,(trY - Y_hat).T**2)
        #backpropagation
        bp_error = -2*W_examples*(trY - Y_hat)*(1-Y_hat**2)
        delta = np.dot(bp_error,trX.T)
        delta_b = np.dot(bp_error, np.ones_like(trY).T)
        W = W - alpha*delta
        b = b - alpha*delta_b
        #error_plot.append(error[0,0])
    
    perceptron_weights.append(W)
    perceptron_biases.append(b)
    beta = 0.5*np.log(np.sum(W_examples*(np.sign(Y_hat) == trY))/np.sum(W_examples*(np.sign(Y_hat) != trY)))
    beta_array.append(beta)
    f_X = f_X + beta*np.sign(Y_hat)
    error = np.sum(np.sign(f_X) != trY)
    error_plot.append(error)
    W_examples_total.append(W_examples)
    W_examples = W_examples*np.exp(-beta*trY*np.sign(Y_hat))

beta_array = np.array(beta_array).reshape(-1,1)
perceptron_weights = np.array(perceptron_weights).reshape(M,2)
perceptron_biases = np.array(perceptron_biases).reshape(M,1)
W_examples_total = np.array(W_examples_total).reshape(M, 160)
""" 
#error plot
plt.title('Missclassified Examples vs. Number of weak learners')
plt.plot(error_plot)
plt.show()
"""

print("Accuracy : " + str(np.sum(np.sign(f_X) == trY)/trY.shape[1]))


#Plot for Adaboost classifiers
#Meshgrid and contour plot drawn similar to the slide
#Got the idea from Shyam for meshgrid

index_metal = np.where(trY == 1)[1]
x_metal = trX[0, index_metal]
y_metal = trX[1, index_metal]

index_rock = np.where(trY == -1)[1]
x_rock = trX[0, index_rock]
y_rock = trX[1, index_rock]

from matplotlib import pyplot as plt
def plotting_function(beta_array, W_examples_total, f_X, perceptron_weights, perceptron_biases):
    x_min, x_max = np.min(trX[0,:]) - 0.01, np.max(trX[0,:]) + 0.01
    y_min, y_max = np.min(trX[1,:]) - 0.01, np.max(trX[1,:]) + 0.01
    x0, x1 = np.meshgrid(np.arange(x_min, x_max, 0.005),
                         np.arange(y_min, y_max, 0.005))
    
    XX = np.c_[x0.ravel(), x1.ravel()].T
    Prediction_contours = np.sum(beta_array*(np.tanh(np.dot(perceptron_weights, XX) + perceptron_biases)), axis = 0)
    
    fig, ax = plt.subplots()
    cs = ax.contourf(x0, x1, Prediction_contours.reshape(x0.shape), 50)
    ax.scatter(x_metal, y_metal, c="y", marker="o", s=20 * W_examples_total[80: ])
    ax.scatter(x_rock, y_rock, c="b", marker="x", s=20 * W_examples_total[:80])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(cs)
    plt.show()

plotting_function(beta_array, W_examples_total, f_X, perceptron_weights, perceptron_biases)