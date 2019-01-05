# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:56:09 2018

@author: Abhilash
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment3\data\data\\"

#load 20 sound waves into a input data matrix of N X 20
data = []
sr1 = 0
for i in range(0,20):
    x, sr1 = librosa.load(base_path + "x_ica_" + str(i+1) + ".wav", sr = None) 
    data.append(np.array(x))

data = np.array(data) # 20 X 76800

#Finding the covariance of the X sample data matrix
X_mean = np.mean(data, axis=1).reshape(-1,1)
X_Expected = np.array(data - X_mean)
data_cov = np.dot(X_Expected, X_Expected.T)  * (1/(X_Expected.shape[1] - 1))


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

"""
ICA - Independent Component Analysis
    W - rotation, or the unmixing matrix that is to be found, initialized with an 4x4 identity matrix
    Z - data in the new reduced dimensional space
    Y - the estimated or the recovered output
    N - Number of samples
"""
delta_W = []
convergence = []
def ICA(Z, rho, N):
    global convergence
    global delta_W
    delta_W= np.identity(4)
    W = np.identity(4)
    Y = np.dot(W,Z)
    for i in range(0, 1000):
        delta_W = np.dot((N*np.identity(4) - np.dot(np.tanh(Y), np.power(Y, 3).T)), W)
        W = W + rho*delta_W
        Y = np.dot(W,Z)
        convergence.append(np.sum(np.abs(delta_W)))
    return W

eigenVectors, eigenValues = PowerIteration(data_cov, 20)
eigenVectors = eigenVectors[:,0:4] #Looking at the eige#n values first 4 are significantly karger than the other values

#Now we scale the eigenvectors to whiten the data with these PCA components
eigenVectors_scaled = np.multiply(eigenVectors, (1/np.sqrt(eigenValues[0:4])))
reduced_data = np.dot(eigenVectors_scaled.T, data)

W = ICA(reduced_data, 0.000001, reduced_data.shape[1])

PCs = np.dot(W, reduced_data)

output_sound = base_path + 'Source1.wav'
librosa.output.write_wav(output_sound,PCs[0,:], sr = sr1)

output_sound = base_path + 'Source2.wav'
librosa.output.write_wav(output_sound,PCs[1,:], sr = sr1)

output_sound = base_path + 'Source3.wav'
librosa.output.write_wav(output_sound,PCs[2,:], sr = sr1)

output_sound = base_path + 'Source4.wav'
librosa.output.write_wav(output_sound,PCs[3,:], sr = sr1)


#Ploting convergence plot
plt.title("Convergence Plot")
plt.plot(convergence)
plt.show()

#Ploting Eigenvalues
plt.title("Eigen Values")
plt.plot(eigenValues)
plt.show()
