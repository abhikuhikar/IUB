#!/usr/bin/env python3
###################################
# CS B551 Fall 2018, Assignment #4
# Abhilash Kuhikar
# Dhruuv Agarwal
# Darshan Shinde

"""
Nearest neighbour:
------------------

In training we just store the X input matrix and labels in the model file.
    
In testing, we have implemented knn by taking Euclidean distance, but without taking the square root 
in the end to same compute time. 

Our code took less than 2 minutes to run in our testing.

KNN testing results for different choices of k
Accuracy k 
0.6871  1 
0.6871  2 
0.7041  3 
0.6967  4
0.7020  5 
0.7104  6
0.7020  7
0.6977  8
0.6967  9
0.7030  10
0.6988  11

The following table is for chosen value of k=6.

train size   accuracy      time (seconds)
7000        0.6542        13.693
12000       0.6723        25.39  
18000       0.6882        30.6118
23000       0.6924        40.566
28000       0.6935        54.821 
32000       0.6988        57.3866  
36000       0.6935        62.468

Our code runs very quickly, thanks to numpy functions like argpartition. 

The compute of the program mainly comes from train size as for test for each example we measure the distance.
So the only free parameter k, doesnt hugely affect time complexity here, it only helps in balancing complexity of the model.
After finding the difference, we just partition the k nearest points, so that depends on how we implement.
Using argpartition we were able to do that in linear time.
         
These are the values for k ranging 1-11 and as we can see the best value we get is at k=6.
This alligns with the notion that we shouldnt have k too low, or too high as the model becomes too complex and too simple respectively in those cases.

As per the question of whether we would suggest this to our clients, KNN is generally used as a baseline as it doesn't need much tuning and is simple to execute.
So we would run knn to get an idea of how much accuracy should we aim for. 
In our current question we got pretty decent run time and accuracy for knn, but as we know knn suffers from few problems like:
    --Slow testing phase- counter-intuitive to machine learning.

 
Adaboost:
---------
    
We have implemented adaboost by using 1 vs 1 technique for multi-class classfiers. Here we have total 4 differnet classes, so in total we have total 6 groups 1 vs 1 classifiers.
Each classifiers have total 555 (observed from number of experiments that gives best accuracy in most cases) random identifed pairs of classifiers.
Classifiers takes filtered data having ground labels matching to the group classifcation labels and classifies the provided input data into 2 classes. 
Test will be performed using all 3330 classfiers and will predicts the labels. Each classifier has a weight assigned to it. We are suming the weights of classifiers predicting same labels.
The label with highest weight will be predicted as the final label for that perticular exampler. 
For 555 classifiers of each group i.e. 3330 in total on training dataset of size 25000, we are getting average accuracy of 70.155 +/- 1.2. It varies because of randomness in feature selection. 
      
adaboost testing results (if complete training dataset is utilized) for different number of classifiers per group
classifiers per group    max testing accuracy achieved after multiple runs
        50                    65.005
        100                   68.187
        150                   69.141
        200                   68.081
        250                   69.141
        300                   70.095
        350                   70.201
        400                   70.201
        450                   70.095
        500                   71.05
        550                   71.262
        600                   69.341
        650                   68.978

We can see that it really gets converge after 300 classifiers of each group and falls after 550 (actually as per my observation it falls after 555).
Hence, we are using 500 (~555) classifiers to get more accurate result. 

adaboost testing results (if 555 classifiers of each group are trained) for different number of classifiers per group
training dataset size    max testing accuracy achieved after multiple runs
        2500                        67.869
        5000                        68.912
        7500                        69.459
       10000                        69.777
       12500                        69.459
       15000                        69.993
       20000                        68.538
       25000                        70.626
       30000                        70.308
       35000                        71.155
       36976                        71.262
       
We can see that it really gets converge after 25000 dataset for training. Although it is still increasing, i think we should limit ourself somewhere 
near to 25000 ~ 26000 to avoid the case of overfitting.

"""
import numpy as np

import random
import time,sys
from collections import Counter
import itertools as it
import matplotlib.pyplot as plt

# Save a dictionary into a pickle file.
import pickle

# Read data from file and generate feature matrix and label list
def readData(filePath):
    input_file = open(filePath,'r')
    image_ids = []
    labels = []
    X= []
    for line in input_file:
        words = line.split()
        X.append(list(map(int, words[2:])))
        image_ids.append(words[0])
        labels.append(int(words[1]))
    X = np.array(X)
    #print(X.shape)
    return X, np.array(labels), np.array(image_ids)

# Read adaboost_model.txt file and generates model
def getAdaboostModelParams(modelfile):
    f = open(modelfile)
    classifiers = []
    for line in f:
        classifiers.append(line.split())
    return classifiers
    
# Training adaboost model
def adaBoostTrain(X, Y):
    X = X.T
    groups = [0,90,180,270]
    groups = list(it.combinations(groups, 2))
    M_trained = []
    for group in groups:
        i = 0
        X_temp = np.array(X[:,np.where(np.logical_or(Y == group[0], Y == group[1]))[0]])
        Y_temp = Y[np.where(np.logical_or(Y == group[0], Y == group[1]))[0]]
        W_examples =  np.ones((1,X_temp.shape[1])) / X_temp.shape[1]
        #for i in range(M):
        while True:
            #Single weak learner
            r1 = np.random.randint(0,X.shape[0]-1)
            r2 = np.random.randint(0,X.shape[0]-1)
            if r1 == r2:
                continue
            stump = X_temp[r1,:] - X_temp[r2,:]
            pos = np.where(stump >= 0)[0]
            neg = np.where(stump < 0)[0]
            Y_hat = stump
            if len(pos) != 0:
                class1 = group[0] if np.sum(W_examples[0,np.where(Y_temp[pos] == group[0])[0]]) \
                                        >= np.sum(W_examples[0,np.where(Y_temp[pos] == group[1])[0]]) else group[1] 
                #class1 = max(set(Y_temp[pos]), key=list(Y_temp[pos]).count)
                Y_hat[Y_hat >= 0] = class1
            if len(neg) != 0:
                class2 = group[0] if np.sum(W_examples[0,np.where(Y_temp[pos] == group[0])[0]]) \
                                        >= np.sum(W_examples[0,np.where(Y_temp[pos] == group[0])[0]]) else group[1] 
                #class2 = max(set(Y_temp[neg]), key=list(Y_temp[neg]).count)
                Y_hat[Y_hat < 0] = class2
            error = np.sum(W_examples[0,np.where(Y_hat != Y_temp)])
            if error >= 0.5 or error == 0:
                continue
            beta =  np.log((1 - error)/error)
            W_examples[:,np.where(Y_hat == Y_temp)] = W_examples[:,np.where(Y_hat == Y_temp)] * error/(1-error)
            total = np.sum(W_examples)
            W_examples = W_examples/total
            m = [r1,r2,class1,class2,beta,group[0],group[1]]
            M_trained.append(m)
            i += 1
            if i >= 555:
                break
    return M_trained

# Write adaboost training model to adaboost_model.txt
def saveAdaboostModelParameter(filepath, params):
    f = open(filepath, 'w')
    for p in params:
        f.write(" ".join(str(x) for x in p))
        f.write("\n")
    f.close()

# Perform adaboost testing on testfile
def testAdaboost(modelfile, X_test, Y_test, testIDs):
    classifiers = getAdaboostModelParams(modelfile)
    predictions = []
    f = open('output.txt', 'w')
    i = 0
    for data in X_test:
        weights = [0,0,0,0]
        for classifier in classifiers:
            r1 = int(classifier[0])
            r2 = int(classifier[1])
            if data[r1] >= data[r2]:
                weights[int(int(classifier[2])/90)] += float(classifier[4])
            else:
                weights[int(int(classifier[3])/90)] += float(classifier[4])
        predictions.append(90 * np.argmax(weights))
        f.write(str(testIDs[i]) + " " + str(predictions[i]) + "\n")
        i += 1
    f.close()
    accuracy = np.round(float(np.sum(predictions == Y_test))/len(Y_test),5)
    print(str(accuracy*100) + "% accuracy")

## used to find the distance from each train data point, and select k nearest points and get the majority label

def knn_calc(X,testxi,k,labelsX):
    inter = np.absolute(X**2-testxi**2)
    inter2 = inter.sum(axis=1)
    ind = np.argpartition(list(inter2), k)[:k]
    predk_group =[labelsX[i] for i in ind]
    orient_dict = Counter(predk_group)
    return max(predk_group, key=orient_dict.get)
## for each test label we call knn_calc
def knn(X,k,labelsX,testX,testY,testIDs):
    i=0
    y=0
    n=0
    preds=[]
    f = open('output.txt', 'w')
    for xi in testX:
        pred_orient = int(knn_calc(X,xi,k,labelsX))
        if pred_orient ==testY[i]: y+=1
        else : n+=1
        
        preds.append(pred_orient)
        f.write(str(testIDs[i]) + " " + str(pred_orient) + "\n")
        i+=1
    
    print(y/(y+n),"knn accuracy with k = 6")        

#takes the input subset X and respective labels Y and outputs best (feature,split) tuple according to the entropy
def bestEntropy(X, Y,features):

    splits = np.random.normal(128,30,10).astype(int)
    entropy_dict = {}
    i=0
    Y=np.array(Y)
    for feature in features:
        node = i
        for split in splits:
            #print( "np.where(X[node] >= split)[0]", np.where(X[node] >= split)[0][0])
            left_label = Y[np.where(X[node] < split)[0]]
            right_label = Y[np.where(X[node] >= split)[0]]
            left_counts_elements = np.unique(left_label, return_counts=True)[1]
            right_counts_elements = np.unique(right_label, return_counts=True)[1]
            entropy = len(left_label)/len(X)*np.sum(-1 *(left_counts_elements/len(left_label)) * np.log(left_counts_elements/len(left_label))) + \
                        len(right_label)/len(X)*np.sum(-1 *(right_counts_elements/len(right_label)) * np.log(right_counts_elements/len(right_label)))
            entropy_dict[(feature,split)] = entropy
        i+=1                 
    return min(entropy_dict, key = entropy_dict.get)

class Node:
    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data
        self.depth = 0
        self.pred = 0

def createTree(X,Y,depth, parent,features):
    Y=np.array(Y)
    #print(len(Y))
    tree = Node(bestEntropy(X,Y,features))
    if parent == None:
        parent = tree
    if parent.depth >= depth:
        return None
    if len(Y)<15 :
        if len(Y)==0: return None 
        a = Counter(Y).most_common(1)
        n= Node(None)
        #print(a)
        n.pred = a[0][0]
        return n
    tree.depth = parent.depth+1
    feature_no = parent.data[0]
    root = features.index(feature_no)
    split = parent.data[1]
    X_left_i = np.where(X[root] < split)[0]
    X_left = X[:,X_left_i]
    Y_left = Y[X_left_i]
    X_right_i = np.where(X[root] >= split)[0]
    X_right = X[:,X_right_i]
    Y_right = Y[X_right_i]
    
    tree.left = createTree(X_left, Y_left, depth, tree,features)
    tree.right = createTree(X_right, Y_right, depth, tree,features)
    if (tree.left==None and tree.right == None):
        a = Counter(Y).most_common(1)
        tree.pred = a[0][0] 
    return tree


def get_subset(X, sub_size,Y=None):
    indices = [random.randint(0,X.shape[0]-1) for i in range(0,sub_size)]
    Xsub = [X[ind] for ind in indices]
    if type(Y)!=type(None):
        Ysub = [Y[ind] for ind in indices]
        return np.array(Xsub),np.array(Ysub)
    return np.array(Xsub),indices


#list_trees = pickle.load( open( "save.txt", "rb" ) )
# Random Forest Algorithm
def r_forest(X, Y, tree_count, ratio):
    list_trees = []
    for i in range(tree_count):
        subset_size = int(X.shape[1]*ratio)
        Xtmp,features = get_subset(X, 20)                     
        Xsub,Ysub = get_subset(Xtmp.T, subset_size,Y)
        Xsub = np.array(Xsub).T
        depth = random.randint(2,5)  
        tree = createTree(Xsub,Ysub, depth, None,features)
        if tree != None:
            list_trees.append(tree)
    return list_trees

def predict_tree(tree, xi):

    if tree.data is None:
        return tree.pred
    if tree.left==None and tree.right==None:
        return tree.pred    
    a = tree.data 
    feature,split = a[0],a[1]
    if xi[feature]<split:
        if tree.left !=None:
            val = predict_tree(tree.left,xi)
        else :
            val = predict_tree(tree.right,xi)
    if xi[feature]>=split:
        if tree.right !=None:
            val = predict_tree(tree.right,xi)
        else:
            val = predict_tree(tree.left,xi)
    return val

def pred_forest(testX, testY,testIDs,list_trees):
    #writing the output to output.txt file
    fp = open('output.txt', 'w')
    predY=[]
    i = 0
    for xi in testX:
        predictions = [predict_tree(tree, xi) for tree in list_trees if tree !=None]
        pred = (Counter(predictions).most_common(1))[0][0]
        predY.append(pred)
        fp.write(str(testIDs[i]) + " " + str(pred) + "\n")
        i+=1
    accuracy = np.sum(predY==testY)/len(testY)
    fp.close()
    return accuracy



#logistic function g
def g(x):
    return 1/(1 + np.exp(-x))

def NN(X,Y):
    #single perceptron training
    alpha = 0.000008
    error_plot = []
    Y_tr = np.array([[Y == 0],[Y == 90],[Y == 180],[Y == 270]]).reshape(4,len(Y))
    
    HiddenUnits = 120
    #Neural Network with input as X i.e. 513x786 matrix. Each input sample is 513 dimensional
    np.random.seed(42)
    #Weights for hidden layer
    W1 = np.random.randn(HiddenUnits,192)
    #biases for hidden layer
    b1 = np.random.randn(HiddenUnits,1)
    #Weights for output layer
    W2 = np.random.randn(4,HiddenUnits) #weights with bias b2
    #biases for output layer
    b2 = np.random.randn(4,1)
    
    epochs = 1200
    for i in range(epochs):
        #hidden layer
        Z1 = np.dot(W1, X.T) + b1
        X2 = g(Z1)
    
        #final layer
        Z2  = np.dot(W2, X2) + b2
        Z2 = Z2/np.sum(np.abs(Z2),axis=0).reshape(1,-1)
        Y_hat = np.exp(Z2) #output of the network 513x786
        Y_hat = Y_hat/np.sum(Y_hat,axis=0).reshape(1,-1)
    
        #backpropagation
        #final layer
        bp_error2 = Y_hat - Y_tr
        delta2 = np.dot(bp_error2,X2.T)
        deltab2 = np.dot(bp_error2,np.ones((bp_error2.shape[1],1)))
        #update weights for output layer
        W2 = W2 - alpha*delta2
        #update bias for output layer
        b2 = b2 - alpha*deltab2
        #hidden layer
        bp_error1 = np.dot(W2.T,bp_error2)*X2*(1-X2)
        delta1 = np.dot(bp_error1, X)
        deltab1 = np.dot(bp_error1, np.ones((bp_error1.shape[1],1)))
        W1 = W1 - alpha*delta1
        b1 = b1 - alpha*deltab1
    
        error = 0.5*sum(sum((Y_hat-Y_tr)**2))
        error_plot.append(error)
    print (error)
    plt.plot(error_plot)
    return W1,b1,W2,b2
    
def NNTest(X1,Y1,testIDs,W1,b1,W2,b2):
    Y_te = np.array([[Y1 == 0],[Y1 == 90],[Y1 == 180],[Y1 == 270]]).reshape(4,len(Y1))
    #prediction on test data
    #hidden layer
    Z1 = np.dot(W1, X1.T) + b1
    X2 = g(Z1)
    #final layer
    Z2  = np.dot(W2, X2) + b2
    Z2 = Z2/np.sum(np.abs(Z2),axis=0).reshape(1,-1)
    Y_hat_test = np.exp(Z2) #output of the network 513x786
    Y_hat_test = Y_hat_test/np.sum(Y_hat_test,axis=0).reshape(1,-1)
    test_accuracy = np.sum(np.argmax(Y_hat_test,axis=0) == np.argmax(Y_te,axis=0))/Y_te.shape[1]
    
    #writing the output to output.txt file
    Y_hat_test = np.argmax(Y_hat_test,axis=0)
    fp = open('output.txt', 'w')
    for i in range(len(Y1)):
        fp.write(str(testIDs[i]) + " " + str(Y_hat_test[i]) + "\n")
    fp.close()

    return test_accuracy

#W1,b1,W2,b2 = NN(X,Y)
#W1,b1,W2,b2 = pickle.load( open( "best_model.txt", "rb" ) )
#NNTest(X1,Y1,W1,b1,W2,b2)
#pickle.dump( [W1,b1,W2,b2], open( "NNModel.txt", "wb" ) )    

if len(sys.argv) < 5:
    print("Usage: \n./orient.py train/test train/test_file.txt modelfile.txt model")
    sys.exit()

(purpose, file,model_data,model) = sys.argv[1:5]
if purpose =="train":
    X,Y,IDs = readData(file)
    data = np.c_[X,Y]
    #print(data)
    if model=="nearest":
        with open(model_data, 'w') as f:
            for row in data:
                f.write(' '.join(map(str,list(row))))
                f.write('\n')
                
            #f.write('# shape: {0}\n'.format(data.shape))
            
        #pickle.dump( data, open( model_data, "wb" ) )
            #np.savetxt(model_data, data,fmt='%d', delimiter='\n',comments='')
    elif model=="adaboost":
        params = adaBoostTrain(X,Y)
        saveAdaboostModelParameter(model_data, params)
    elif model=="forest":
        params = r_forest(X.T, Y, 800, 0.04)
        pickle.dump( params, open( model_data, "wb" ) ) 
    else:
        params = NN(X,Y)
        pickle.dump( params, open( model_data, "wb" ) ) 
        
if purpose =="test":
    #inp = open(model_data,'r')
    testX,testY,testIDs = readData(file) #test-data.txt #custom-test.txt
    
    t1 = time.time()
    if model=="nearest":
        line = ""
        input_file = open(model_data,'r')
        Xdata= []
        for line in input_file:
            words = line.split()
            Xdata.append(list(map(int, words)))
        Xdata = np.array(Xdata)
                
        X= Xdata[:,:-1]
        Y= Xdata[:,-1]
        knn(X,6,Y,testX,testY,testIDs)
        print(" knn test time:",time.time()-t1)
    elif model=="adaboost":
        testAdaboost(model_data, testX, testY, testIDs)
    elif model=="forest":
        list_trees = pickle.load( open( model_data, "rb" ) )
        accuracy = pred_forest(testX,testY, testIDs,list_trees)
        print("Accuracy for forest = " + str(accuracy*100) + "%")
    else:
        W1,b1,W2,b2 = pickle.load( open( model_data, "rb" ) )
        accuracy = NNTest(testX,testY,testIDs,W1,b1,W2,b2)
        print("Accuracy for Neural Network(best) = " + str(accuracy*100) + "%")

    
        
#M_trained = adaBoost(X,Y)
