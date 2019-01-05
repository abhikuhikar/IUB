# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 02:03:40 2018

@author: Abhilash
"""
#Tasks included in this file:
#Power iteration routine that calculates the eigenvector one by one
#Load flut.mat and plot this using a color map
#Calculate covariance of a matrix
#Learn two eigenvectors from 513x513 covariance matrix
#We have got representative spectra for our two flute notes
#Plot your eigenvectors and put them on your report along with the X matrix.
#How would you recover their temporal activations?
#You need to show me how to calculate these activation vectors in equations.
#Plot the activation (row) vectors you got from this procedure in the report.
#Perform power iteration twice to get the two eigenvectors.
#They should correspond to the temporal activations you calculated in the pre-vious question,
# but this time you got them in a dierent way.
#How would you get the representative spectra this time?
#Out of the two approaches, i.e. doing eigendecomposition on the original
    #data matrix and its transposed version, which do you prefer? Explain
    #your preference. I need a clear reason.
#temporal activation for the second case will be the representative spectra for hte first case

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

filePath = "F:\Fall18_Sem1\MLSP\Assignments\Assignment1\AudioFiles\\"

fileName = filePath + 'flute.mat'

fluteSignal = scipy.io.loadmat(fileName)

X_flut = fluteSignal['X']

#Plot code refered from Shyam Narasimhan
plt.figure(figsize = (1,4))
plt.title('Original flute matrix graph')
plt.imshow(X_flut, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#Finding the covariance of the X matrix
X_mean = np.mean(X_flut, axis=1).reshape(-1,1)
X_Expected = np.array(X_flut - X_mean)
X_Covariance = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))

#Power Iteration
def PowerIteration(X, num_EigenVectors):
    eigen_vectors = []
    for i in range(0,num_EigenVectors):
        Y = np.random.rand(X.shape[0], 1)
        for j in range(0,1000):
            Y = np.dot(X, Y)
            Y = Y * (1/np.power(np.sum(np.power(Y,2)),0.5))
        singular_value = np.power(np.sum(np.power(np.dot(Y.T, X), 2)), 0.5)
        right_singular_vector = np.dot(X.T, Y) * (1/singular_value)
        print(str(singular_value) + "Singular value")
        residual_vector = singular_value * np.dot(Y, right_singular_vector.T)
        X = X - residual_vector
        eigen_vectors.append(Y.reshape(-1,))
        
    return eigen_vectors

#eigen vectors for 513x513 covariance matrix
eigenVectors = (np.array(PowerIteration(X_Covariance, 2))).T 

#Calculating the temporal activations
temporal_activation = np.dot(eigenVectors.T, X_flut)

#Calculating another covariance matrix considering the data samples: 73x73
#Finding the covariance of the X matrix
X_flut_tr = X_flut.T
X_mean = np.mean(X_flut_tr, axis=1).reshape(-1,1)
X_Expected = np.array(X_flut_tr - X_mean)
X_Covariance2 = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))

#eigen vectors for 73x73 covariance matrix
eigenVectors2 = (np.array(PowerIteration(X_Covariance2, 2))).T #eigen vectors for 513x513 covariance matrix

#Calculating the temporal activations
temporal_activation2 = np.dot(eigenVectors2.T, X_flut.T)

#Calculating rescontrsuction error by revoering the original data
X_flut_recover1 = np.dot(eigenVectors, temporal_activation)
X_flut_recover2 = (np.dot(eigenVectors2, temporal_activation2)).T


error1 = np.sqrt(np.sum(np.power((X_flut - X_flut_recover1), 2)))

error2 = np.sqrt(np.sum(np.power((X_flut - X_flut_recover2), 2)))

#Plot code refered from Shyam Narasimhan
plt.figure(figsize = (1,4))
plt.title('Eigen Vectors')
plt.imshow(eigenVectors, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()


#Plot code refered from Shyam Narasimhan
plt.figure(figsize = (4,1))
plt.title('Activation1')
plt.imshow(temporal_activation, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#Eigen Vectors 2
plt.figure(figsize = (1,4))
plt.title('EigenVectors2')
plt.imshow(eigenVectors2, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#Eigen Vectors 2 transpose
plt.figure(figsize = (4, 1))
plt.title('EigenVectors2')
plt.imshow(eigenVectors2.T, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#temporal activation 2
plt.figure(figsize = (1, 4))
plt.title('ReprSpectra')
plt.imshow(temporal_activation2.T, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#recovery 1
plt.figure(figsize = (1, 4))
plt.title('Recovery1')
plt.imshow(X_flut_recover1, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#recovery 2
plt.figure(figsize = (1, 4))
plt.title('Recovery2')
plt.imshow(X_flut_recover2, cmap='jet', interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()
