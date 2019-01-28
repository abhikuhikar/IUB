#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print (im.size)
    print (int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


def toMatrix(l):
    mat = np.zeros([CHARACTER_HEIGHT, CHARACTER_WIDTH])
    for i in range(len(l)):
        for j in range(CHARACTER_WIDTH):
            mat[i][j] = l[i][j] == '*'
    return mat

def emission2(test, label):
    m = 0
    if np.count_nonzero(label) == 0:
        if np.count_nonzero(test) <= 5:
            return 1e-60
        else:
            return 1e-100
    else:
        for i in range(CHARACTER_HEIGHT):
            for j in range(CHARACTER_WIDTH):
                if label[i][j] == 1 and label[i][j] == test[i][j]:
                    m += 1
    return m/np.count_nonzero(label)

def error(test):
    bestLabel = ground_labels[maxEmission(test)]
    err = (np.size(test) - np.count_nonzero(test == train_letters_mat[bestLabel]))/np.size(test)
    return err
#gives the emission probability : P(ObsLetter|hiddenLetter) = P(test|label)
def emission(test, label):
    m = 0
    N = np.size(label)
    p = 0.32
    if np.count_nonzero(label) == 0:
        if np.count_nonzero(test) <= 5:
            return 1e-60
        else:
            return 1e-78
    else:
        m = np.sum(test == label)
    e = np.power(1-p,m)*np.power(p,N-m)
    return e
    #return m/np.count_nonzero(label)

#gives the max emission probablity for a given observed letter.
#It also gives the hidden label index corresponding to the max value
def maxEmission(t):
    probs = [emission2(t, train_letters_mat[l]) for l in ground_labels]
    return np.argmax(probs)

def read_data(fname):
    letters = []
    file = open(fname, 'r');
    for line in file:
        letters += list(line)
    letters = list(filter(lambda i: i in list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "), letters))
    return letters

def train(letters):
    global char_init_probs, transition, transitionMatrix
    prevs = ""
    transition = {key1 : {key2 : 1e-100 for key2 in ground_labels} for key1 in ground_labels}
    for s in letters:
        if prevs == "":
            char_init[s] = char_init.get(s, 0) + 1
            prevs = s
        else:
            #print(transition)
            if prevs == " ":  char_init[s] = char_init.get(s, 0) + 1
            transition[prevs][s] += 1
            prevs = s
    for s1 in transition:  ## modelled as new s or Si+1|Si so normalized on prev s or Si
        total = sum(transition[s1].values())
        for s2 in transition[s1]:
            transition[s1][s2] = transition[s1][s2]/total
    char_init_probs = {key : 1e-100 for key in ground_labels}
    for s in char_init.keys():  ## generate the initial probabilities
        char_init_probs[s] = char_init[s]/sum(char_init.values())
    for key1 in transition:
        for key2 in transition[key1]:
            transitionMatrix[ground_labels.index(key1)][ground_labels.index(key2)] = transition[key1][key2]				

def simple():
    sequence = []
    for t in test_letters_mat:
        emission_probs = [emission(t, train_letters_mat[l]) for l in ground_labels]
        index = np.argmax(np.array(np.log(emission_probs)) + np.log(np.array(list(char_init_probs.values()))))
        sequence.append(ground_labels[index])
    return sequence

    
def hmm_viterbi(sentence):
    #for w,s in sentence.
    #construct a transition table storing the values
    V = np.zeros((72,len(sentence)))
    i = 0
    backtrack = np.zeros(V.shape)
    for c in sentence:        
        #initial probabilities
        if i == 0:
            for char in ground_labels:
                emission_prob = emission(c,train_letters_mat[char])
                V[ground_labels.index(char),i] = np.log(char_init_probs[char]) + np.log(emission_prob)
        else:
            #transition to next state
            for char in ground_labels:
                emission_prob = emission(c,train_letters_mat[char])
                vector = np.log(transitionMatrix.T[ground_labels.index(char),:].reshape(1,-1)) + \
                            V[:,i-1].reshape(1,-1)
                best = np.argmax(vector)
                backtrack[ground_labels.index(char),i] = best
                V[ground_labels.index(char),i] = np.max(vector) + np.log(emission_prob)
        i += 1
    #backtracking
    prev_best = np.argmax(V[:,-1])
    best_sequence = [ground_labels[prev_best]]
    for i in range(len(sentence)-1):
        index = len(sentence) - i - 1
        prev_best = int(backtrack[prev_best,index])
        best_sequence.append(ground_labels[prev_best])
    best_sequence.reverse()
    return best_sequence

#####
# main program
def test():
    global test_letters, test_letters_mat
    for i in range(19):
        test_img_fname = "test-" + str(i) + "-0.png"
        test_letters = load_letters(test_img_fname)
        test_letters_mat = [toMatrix(l) for l in test_letters]
        emission_prob = [maxEmission(test_letters_mat[i]) for i in range(len(test_letters_mat))]
        sequence = [ground_labels[emission_prob[i]] for i in range(len(test_letters_mat))]
        print (''.join(sequence))
        sequence2 = simple()
        print (''.join(sequence2))
        sequence3 = hmm_viterbi(test_letters_mat)
        print (''.join(sequence3))

test()
## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print ("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print ("\n".join([ r for r in test_letters[2] ]))
