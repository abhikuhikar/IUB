# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:55:52 2018

@author: hp-pc
"""
import sys
import string

def train_test_data(train_filename,test_filename):
    #opening files
    inp_train = open(train_filename,"r",encoding = "utf-8")
    inp_list_train = [line for line in inp_train.read().split("\n")]
    inp_test = open(test_filename,"r",encoding = "utf-8")
    inp_list_test = [line for line in inp_test.read().split("\n")] 
    #closing files
    inp_train.close()
    inp_test.close()
    #handling empty lines
    lines_train = [line for line in inp_list_train if len(line) is not 0]
    words_train = [[word for word in line.split()] for line in lines_train]

    lines_test = [line for line in inp_list_test if len(line) is not 0]
    words_test = [[word for word in line.split()] for line in lines_test]
    #handling non ascii characters
    words_test = [["".join(letter for letter in word if ord(letter)<128) for word in line]\
               for line in words_test]

    labels = ["Los_Angeles,_CA","San_Francisco,_CA","Manhattan,_NY","Houston,_TX","Washington,_DC",\
              "Toronto,_Ontario","Boston,_MA","Chicago,_IL","Philadelphia,_PA","Atlanta,_GA",\
              "Orlando,_FL","San_Diego,_CA"]

    #handling multiple lines in training data
    j = -1
    words_new_train = []
    for i in range (0,len(words_train)):
        if (len(words_train[i])>0 and words_train[i][0] in labels):
            words_new_train.append(words_train[i])
            j=j+1
        else:
            words_new_train[j] = words_new_train[j] + words_train[i]

    #handling multiple lines in testing data        
    j = -1
    words_new_test = []
    tweet_list = []
    for i in range (0,len(words_test)):
        if (len(words_test[i])>0 and words_test[i][0] in labels):
            words_new_test.append(words_test[i])
            tweet_list.append([" ".join(words_test[i][1:])])
            j=j+1
        else:
            words_new_test[j] = words_new_test[j] + words_test[i] 
     
       
    trivial_words = ['I','ON','on','On','this','This','A','The','An','For',\
                    'an','a','at','for','of','to','in','and','is','the']    
    #extract training labels
    ytrain = [word[0] for word in words_new_train]
    #removing punctuations
    words_new_train=[["".join(letter for letter in word if letter not in string.punctuation).lower()\
                      for word in line if word not in trivial_words ]for line in words_new_train]
    #removing "" characters
    words_new_train = [[word for word in line if word is not ""]for line in words_new_train]
    #seperating list of training words
    Xtrain = [word[1:] for word in words_new_train]

    #extract testing labels
    ytest = [word[0] for word in words_new_test]
    #removing punctuations
    words_new_test=[["".join(letter for letter in word if letter not in string.punctuation).lower()\
                     for word in line if word not in trivial_words]for line in words_new_test]
    #removing "" characters
    words_new_test = [[word for word in line if word is not ""]for line in words_new_test]
    #seperating list of testing words
    Xtest = [word[1:] for word in words_new_test]
    return Xtrain,Xtest,ytrain,ytest,tweet_list,labels  

def data_dict(Xtrain,ytrain):    
    #Creating dictionary{city-->word-->word_count}
    doc_dict = {}
    for i in range(0, len(Xtrain)):
        city = ytrain[i]
        if city not in doc_dict.keys():
            doc_dict[city] = {'freq':1,'word_freq':0}
            for j in range(0,len(Xtrain[i])):
                word = Xtrain[i][j]
                if word not in doc_dict[city].keys():
                    doc_dict[city][word] = 1
                else:
                    doc_dict[city][word] += 1
                doc_dict[city]['word_freq'] += 1
        else:
            doc_dict[city]['freq'] += 1
            for j in range(0,len(Xtrain[i])):
                word = Xtrain[i][j]
                if word not in doc_dict[city].keys():
                    doc_dict[city][word] = 1
                else:
                    doc_dict[city][word] += 1
                doc_dict[city]['word_freq'] += 1 
    return doc_dict                   

def cond_prob(X,word,city):
    #calculates P(a|b) where a:word, b:city
    #X is dictionary {city-->words-->word_count}
    if word not in X[city].keys():
        return 0.0000001
    else:
        return X[city][word]/X[city]['word_freq']

def Naive_Bayes(tweets,data):
    predicted = []
    word_prob_dict = {}
    for tweet in tweets:
        pred = {"Los_Angeles,_CA":0,"San_Francisco,_CA":0,"Manhattan,_NY":0,"Houston,_TX":0,"Washington,_DC":0,\
            "Toronto,_Ontario":0,"Boston,_MA":0,"Chicago,_IL":0,"Philadelphia,_PA":0,"Atlanta,_GA":0,\
            "Orlando,_FL":0,"San_Diego,_CA":0}    
        for city in pred.keys(): 
            prob = 1
            if city not in word_prob_dict.keys():
                word_prob_dict[city] = {}
            for word in tweet:
                c_prob = cond_prob(data,word,city)
                prob = prob * c_prob
                if word not in word_prob_dict[city].keys():
                     word_prob_dict[city][word] = c_prob   
            pred[city] = prob * data[city]['freq']/32000
        predicted.append(max(pred,key=pred.get))
    return predicted, word_prob_dict

          
#taking inputs from command line
train = sys.argv[1]
test = sys.argv[2]
out = sys.argv[3]
  
#modelling training and testing data
Xtrain,Xtest,ytrain,ytest,tweet_list,labels = train_test_data(train,test)
#making dictionary{city-->words-->word_count}
data = data_dict(Xtrain,ytrain)
#running classifier
y_pred, word_dict = Naive_Bayes(Xtest,data)

#output to be written in output file
output_file = open(out,"w",encoding = "utf-8")
for i in range(0,len(y_pred)):
    output_file.write(y_pred[i] + ',' + ytest[i] + ',' + "".join(tweet_list[i])+"\n")
output_file.close()        

#printing 5 most common words for each city
top_words = []
for city in labels:
    x = sorted(((v,k) for k,v in word_dict[city].items()), reverse=True)
    top_words.append((city,x[:5]))
print("Top 5 common words in each city: ")
for i in top_words:
    print(i[0],":",i[1][0][1],",",i[1][1][1],",",i[1][2][1],",",i[1][3][1],",",i[1][4][1])   
