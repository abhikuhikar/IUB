# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:53:41 2018

@author: Abhilash
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment4\data\data\\"

concentric = scipy.io.loadmat(base_path + 'concentric.mat')
concentricX = concentric['X']

def RBFKernel(xi, xj):
    dist = np.sum(np.power((xi - xj), 2))
    return np.exp(-dist/0.01)

#Power Iteration
def PowerIteration(X, num_EigenVectors):
    eigen_vectors = []
    eigen_values = []
    for i in range(0,num_EigenVectors):
        Y = np.random.rand(X.shape[0], 1)
        for j in range(0,1000):
            Y = np.dot(X, Y)
            Y = Y * (1/np.power(np.sum(np.power(Y,2)),0.5))
        singular_value = np.power(np.sum(np.power(np.dot(Y.T, X), 2)), 0.5)
        eigen_values.append(singular_value)
        right_singular_vector = np.dot(X.T, Y) * (1/singular_value)
        residual_vector = singular_value * np.dot(Y, right_singular_vector.T)
        X = X - residual_vector
        eigen_vectors.append(Y.reshape(-1,))
    eigen_vectors = np.array(eigen_vectors)
    return eigen_vectors.T, eigen_values

kernel_PCA = np.array([np.power(concentricX[0,:],2), np.power(concentricX[1,:],2), concentricX[0,:]*concentricX[1,:], np.ones(concentricX.shape[1])])
kernel_X = np.array([[RBFKernel(concentricX[:,i], concentricX[:,j]) for i in range(concentricX.shape[1])] for j in range(concentricX.shape[1])])
eigenvectors, eigenvalues = PowerIteration(kernel_X, 3)
W = np.random.uniform(0,1,size=4).reshape(-1,1)
eigenvectors = eigenvectors.T
#logistic function g
def g(x):
    return 1/(1 + np.exp(-x))

#learning rate alpha4
alpha = 0.05

Y = np.array([int(i > 50) for i in range(152)])
error_plot = []
X = np.concatenate((eigenvectors, np.ones(eigenvectors.shape[1]).reshape(1,-1)), axis=0)
for i in range(80000):
    Z = np.dot(W.T, X)
    Y_hat = g(Z)
    error = 0.5*np.dot((Y-Y_hat),(Y-Y_hat).T)
    error_plot.append(error)
    #to make the labels for "i < 51" 0
    g_dash = (Y_hat*(1 - Y_hat))
    delta_W = np.dot(X,((Y_hat - Y)*g_dash).T)
    W = W - alpha*delta_W

Y_hat = g(np.dot(W.T, X))
error = 0.5*np.dot((Y-Y_hat),(Y-Y_hat).T)
print (error)
error_plot = np.array(error_plot).reshape(80000,1)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs = X[0,:], ys = X[1,:], zs=X[2,:], c=None)

# create x,y
xx, yy = np.meshgrid(range(-1,1), range(-1,1))
z = (-W[0] * xx - W[1] * yy - W[3]) * 1. /W[2]
# plot the surface
ax = plt.gca()
#ax.hold(True)
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.2)
ax.plot_surface(xx, yy, z, alpha=0.2)


#plot 2d scatter plot
plt.figure(figsize= (5,4))
plt.title('Scatter plot on 2D space')
plt.scatter(concentricX[0,:], concentricX[1,:],)
plt.show()

#error plot
plt.title('Error plot')
plt.plot(error_plot)
plt.show()
