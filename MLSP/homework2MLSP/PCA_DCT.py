# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:40:43 2018

@author: Abhilash
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment2\data\data\\"

x_source = base_path + "s.wav"

Original_Source, sr1 = librosa.load(x_source, sr=None)

N = 1024

#Create data matrix with consecutive random sampling with N colun vectors 
def createSampleDataMatrix(source, N):
    start_index = np.random.randint(low = 0, high = source.shape[0] - 8, size=N)
    sampleDataMatrix = []
    for i in range(0, N):
        sample_row = np.array(source[start_index[i]:8 + start_index[i]])
        sampleDataMatrix.append(sample_row)
    
    sampleDataMatrix = np.array(sampleDataMatrix).T
    return sampleDataMatrix

#Finding the covariance of the X sample data matrix
def CovarianceMatrix(matrix):
    X_mean = np.mean(matrix, axis=1).reshape(-1,1)
    X_Expected = np.array(matrix - X_mean)
    matrix_Cov = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))
    
    return matrix_Cov

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
        residual_vector = singular_value * np.dot(Y, right_singular_vector.T)
        X = X - residual_vector
        eigen_vectors.append(Y.reshape(-1,))
    print()
    eigen_vectors = np.array(eigen_vectors)
    return eigen_vectors.T

X_SampleDataMatrix_10 = createSampleDataMatrix(Original_Source, 10)

#Finding the covariance of the X sample data matrix
X_mean = np.mean(X_SampleDataMatrix_10, axis=1).reshape(-1,1)
X_Expected = np.array(X_SampleDataMatrix_10 - X_mean)
X_SampleDataMatrix_10_cov = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))

Eigen_Vectors_10 = PowerIteration(X_SampleDataMatrix_10_cov, 8)


#Repeating the procedure for N = 100
X_SampleDataMatrix_100 = createSampleDataMatrix(Original_Source, 100)

#Finding the covariance of the X sample data matrix
X_mean = np.mean(X_SampleDataMatrix_100, axis=1).reshape(-1,1)
X_Expected = np.array(X_SampleDataMatrix_100 - X_mean)
X_SampleDataMatrix_100_cov = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))
Eigen_Vectors_100 = PowerIteration(X_SampleDataMatrix_100_cov, 8)

#Repeating the procedure for N = 1000
X_SampleDataMatrix_1000 = createSampleDataMatrix(Original_Source, 1000)

#Finding the covariance of the X sample data matrix
X_mean = np.mean(X_SampleDataMatrix_1000, axis=1).reshape(-1,1)
X_Expected = np.array(X_SampleDataMatrix_1000 - X_mean)
X_SampleDataMatrix_1000_cov = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))

Eigen_Vectors_1000 = PowerIteration(X_SampleDataMatrix_1000_cov, 8)

#EigenVectors with 10 Samples
plt.figure(figsize= (5,4))
plt.title('EigenVectors with 10 Samples')
plt.imshow(Eigen_Vectors_10.T, interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#EigenVectors with 100 Samples
plt.figure(figsize= (5,4))
plt.title('EigenVectors with 100 Samples')
plt.imshow(Eigen_Vectors_100.T, interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()

#EigenVectors with 1000 Samples
plt.figure(figsize= (5,4))
plt.title('EigenVectors with 1000 Samples')
plt.imshow(Eigen_Vectors_1000.T, interpolation='none', aspect ='auto')
plt.axis('off')
plt.show()
