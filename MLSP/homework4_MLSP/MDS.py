# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:01:11 2018

@author: Abhilash
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment4\data\data\\"

mds_mat = scipy.io.loadmat(base_path + 'MDS_pdist.mat')

mds_pdist = mds_mat['L']

m_tilda = mds_pdist - np.mean(mds_pdist, axis=1).reshape(-1,1)
W = m_tilda - np.mean(m_tilda, axis=0)

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

eigenvectors, eigenvalues = PowerIteration(W,2)
lambda_vect = np.diag(eigenvalues)
recovered_mat = np.dot(eigenvectors, np.sqrt(lambda_vect))
plt.scatter(recovered_mat[:,0], recovered_mat[:,1], )

#Plot
plt.title('MDS recovered map')
plt.scatter(recovered_mat[:,0], recovered_mat[:,1], )
plt.show()

#Plot
plt.title('MDS rotated recovered map')
plt.scatter(-recovered_mat[:,1], recovered_mat[:,0], )
plt.show()