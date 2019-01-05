# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 01:52:41 2018

@author: Abhilash
"""

from scipy import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment2\data\data\\"

june_mat_file = base_path + 'june.mat'
june_mat_load = io.loadmat(june_mat_file)
june_matrix = june_mat_load['june']

dec_mat_file = base_path + 'december.mat'
dec_mat_load = io.loadmat(dec_mat_file)
december_matrix = dec_mat_load['december']

disparity_matrix = (december_matrix[:,0] - june_matrix[:,0]).reshape(-1, 1)

def k_Means_Clustering(data, number_clusters):
    clusters = np.random.uniform(low = np.min(data), high = np.max(data), size=number_clusters)
    
    membership_matrix = np.zeros(len(data)*number_clusters).reshape(len(data), number_clusters)
    
    for iter in range(0,900):
        #Assign clusters to the data points
        for i in range(0, len(data)):
            if (np.abs(data[i] - clusters[0])) < (np.abs(data[i] - clusters[1])):
                membership_matrix[i][0] = 1
                membership_matrix[i][1] = 0
            else:
                membership_matrix[i][1] = 1
                membership_matrix[i][0] = 0

        clusters = (np.dot(data.T, membership_matrix) / np.sum(membership_matrix, axis = 0)).reshape(-1)
    
    
    #Start : Code for ploting data frame suggested by Shyam Narasimhan
    membership_matrix_column = np.argmax(membership_matrix, axis = 1)
    data_frame = pd.DataFrame(data)
    data_frame.columns = ['Disparity Data']
    data_frame['Cluster'] = membership_matrix_column
    #End
    
    return clusters, data_frame

clusters, data_frame = k_Means_Clustering(disparity_matrix, 2)

#Ploting the disparity matrix
plt.figure(figsize= (6,4))
plt.title("Disparity Matrix histogram")
plt.hist(disparity_matrix)
plt.show()

#Start : Code for ploting data frame suggested by Shyam Narasimhan
import seaborn as sns
sns.countplot(data_frame['Disparity Data']).set_title("Disparity Data Histogram")
sns.countplot(data_frame['Disparity Data'], hue = data_frame['Cluster']).set_title("K-Means Clustering")
#end