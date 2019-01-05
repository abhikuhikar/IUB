# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 00:06:49 2018

@author: Abhilash
"""

from scipy import io
import numpy as np
import scipy.stats
import pandas as pd

base_path = "F:\Fall18_Sem1\MLSP\Assignments\Assignment2\data\data\\"

june_mat_file = base_path + 'june.mat'
june_mat_load = io.loadmat(june_mat_file)
june_matrix = june_mat_load['june']

dec_mat_file = base_path + 'december.mat'
dec_mat_load = io.loadmat(dec_mat_file)
december_matrix = dec_mat_load['december']

parity_matrix = (december_matrix[:,0] - june_matrix[:,0]).reshape(-1, 1)

def normal_pdf(mu, sigma, x):
    return (1/np.sqrt((2*np.pi*sigma*sigma)))*(np.exp(-((x-mu)*(x-mu)/(2*sigma*sigma))))

def EM_GMM(data, num_gaussians):
    P_A = 0.5
    P_B = 0.5
    Mean_A = 37
    Sigma_A = 2
    Sigma_B = 3
    Mean_B = 63
    for iter in range(0,500):
        P_X_Given_A = normal_pdf(Mean_A, Sigma_A, data)
        P_X_Given_A[P_X_Given_A==0] = 0.05
        P_X_Given_B = normal_pdf(Mean_B, Sigma_B, data)
        P_X_Given_B[P_X_Given_B==0] = 0.05
        
        U_A = P_A*P_X_Given_A/(P_A*P_X_Given_A + P_B*P_X_Given_B)
        U_B = P_B*P_X_Given_B/(P_A*P_X_Given_A + P_B*P_X_Given_B)
        
        Mean_A = np.sum(U_A*data)/(np.sum(U_A))
        Mean_B = np.sum(U_B*data)/(np.sum(U_B))
        
        P_A = np.mean(U_A)
        P_B = np.mean(U_B)
        
        Var_A = np.sum(U_A*(data - Mean_A)*(data - Mean_A))/np.sum(U_A)
        Sigma_A = np.sqrt(Var_A)

        Var_B = np.sum(U_B*(data - Mean_B)*(data - Mean_B))/np.sum(U_B)
        Sigma_B = np.sqrt(Var_B)
    
    #Start : Code for ploting data frame suggested by Shyam Narasimhan
    membership_matrix = np.argmax(np.concatenate((U_A, U_B), axis = 1), axis = 1)
    data_frame = pd.DataFrame(data)
    data_frame.columns = ['Disparity Data']
    data_frame['Cluster'] = membership_matrix
    #end
    
    return Mean_A, Mean_B, Sigma_A, Sigma_B, data_frame
    

Mean_A, Mean_B, Sigma_A, Sigma_B, data_frame = (EM_GMM(parity_matrix, 2))

#Start : Code for ploting data frame suggested by Shyam Narasimhan
import seaborn as sns
sns.countplot(data_frame['Disparity Data'], hue = data_frame['Cluster'])
#end