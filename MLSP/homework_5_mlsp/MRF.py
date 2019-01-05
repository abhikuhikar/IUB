# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:50:38 2018

@author: Abhilash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import scipy 
base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment5\data\data\\"

im0 = Image.open(base_path + "im0.ppm")
im8 = Image.open(base_path + "im8.ppm")
X_L = np.array(im0, dtype=int)
X_R = np.array(im8)

#Numbwe of guassians
NUM_G = 4
def nearestPixel(i,j):
    nearest = 0
    right = X_R[i,j]
    minDist = 1000000
    for k in range(40):
        left =  X_L[i,j+k]
        dist = sum(np.abs(right - left))
        if dist < minDist:
            
            minDist = dist
            nearest = k
    return nearest

#Disparity mao
D = np.zeros([X_L.shape[0],X_L.shape[1] - 40])

for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        D[i,j] = nearestPixel(i,j)


D_vector = D.reshape(-1,1)

plt.hist(D_vector, bins = 40)

def normal_pdf(mu, sigma, x):
    return (1/np.sqrt((2*np.pi*sigma*sigma)))*(np.exp(-((x-mu)*(x-mu)/(2*sigma*sigma))))

#GMM Clustering
def EM_GMM(data, num_gaussians):
    P_Cluster = np.array([1/num_gaussians]*num_gaussians)
    np.random.seed(50)
    ClusterMeans = np.random.uniform(10, 50, num_gaussians)
    ClusterSDs = np.random.uniform(1, 5, num_gaussians)
    P_X_Given_Cluster = np.zeros((np.size(data), num_gaussians))
    U_Cluster = np.zeros((np.size(data), num_gaussians))
    for iter in range(0,500):
        #E-Step
        for i in range(num_gaussians):
            P_X_Given_Cluster[:,i] = normal_pdf(ClusterMeans[i], ClusterSDs[i], data).reshape(1,-1)
            P_X_Given_Cluster[P_X_Given_Cluster==0] = 0.00005
            
        P_X = np.sum(P_Cluster*P_X_Given_Cluster, axis=1)
        U_Cluster = P_Cluster*P_X_Given_Cluster/P_X.reshape(-1,1)
        
        #M-Step
        ClusterMeans = np.sum(U_Cluster*data.reshape(-1,1), axis = 0)/np.sum(U_Cluster, axis = 0)
        P_Cluster = np.mean(U_Cluster, axis=0)
        for i in range(num_gaussians):
            ClusterSDs[i] = np.sqrt(np.sum(U_Cluster[:,i].reshape(-1,1)*((data - ClusterMeans[i])*\
                      (data - ClusterMeans[i])))/np.sum(U_Cluster[:,i]))
        
    membership_matrix = np.argmax(U_Cluster, axis = 1)
    data_frame = pd.DataFrame(data)
    data_frame.columns = ['Disparity Data']
    data_frame['Cluster'] = membership_matrix
    #end
    
    return ClusterMeans, ClusterSDs, data_frame

ClusterMeans, ClusterSDs, data_frame = EM_GMM(D_vector, NUM_G)
cluster_map= np.array(data_frame['Cluster'])
depth_map = np.array(data_frame['Cluster'], dtype=float)
for i in range(NUM_G):
    depth_map[depth_map == i] = ClusterMeans[i]

depth_map = depth_map.reshape(D.shape)
cluster_map = cluster_map.reshape(D.shape)
plt.imshow(depth_map,cmap="gray")
#import seaborn as sns
#sns.countplot(data_frame['Disparity Data'], hue = data_frame['Cluster'])

#for Gibbs sampling
C = []
smoothened_cluster_map = np.array(cluster_map)
prev_cluster_map = np.array(cluster_map)
smoothened_depth_map = np.zeros(D.shape)

#similarity
def similarity(i, j, cluster):
    if i < 0 or j < 0 or i > prev_cluster_map.shape[0]-1 or j > prev_cluster_map.shape[1]-1\
        or prev_cluster_map[i,j] == cluster:
        return 1
    a = 5
    sigma = 0.5
    if prev_cluster_map[i,j] == cluster:
        a = 0
    return np.exp(-(a*a/(2*sigma*sigma)))

#prior probability based on 8-neighboring nodes of i,j whose cluster is 'cluster'
def prior(i,j,cluster):
    N = [-1,0,1]
    prior_p = 1
    for k in N:
        for l in N:
            prior_p *= similarity(i+k,j+l, cluster)
    return prior_p

#Gibbs Sampling
for iter in range(30):
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            #change the label at i,j
            #calculate 4 cluster probability values for the given cluster
            #let's say i,j belongs to 2nd cluster
            #P(C|X) = P(X|C)*markov_similarity
            current_cluster = cluster_map[i,j]
            posterior = np.zeros(NUM_G)
            for k in range(NUM_G):
                posterior[k] = normal_pdf(ClusterMeans[k], ClusterSDs[k], depth_map[i,j]) * prior(i,j,k)
            posterior = posterior/np.sum(posterior)
            new_label = np.random.choice(np.arange(0, NUM_G), p=posterior)
            smoothened_depth_map[i,j] = ClusterMeans[new_label]
            smoothened_cluster_map[i,j] = new_label
    C.append(smoothened_depth_map)
    prev_cluster_map = np.array(smoothened_cluster_map)
        #calculate P(1st cluster| i,j), P(2nd cluster| i,j) , 3rd, 4th, .... using M10 S18
        #now sample according to these probabilities and replace the sample
        #repeat for all the pixels
        #repeat this entire procedure for some iterations and take the last few samples to take the count

plt.imshow(smoothened_depth_map,cmap="gray")
C = np.array(C)
#majority voting
smoothened_final_map = scipy.stats.mode(C[-10:])[0][0]
plt.imshow(smoothened_final_map,cmap="gray")

#for i in range(smoothened_depth_map.shape[0]):
#    for j in range(smoothened_depth_map.shape[1]):
#        smoothened_final_map[i,j] = scipy.stats.mode(C[-10:, i, j])[0][0]


