###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
#   Dhruuv Agarwal : 2000246422

 
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    pos_init = {}
    pos_init_probs = {}
    transition = {}
    emission = {}
    pos = {}
    mapPOS = {}
    inv_mapPOS = {}
    transitionMatrix = np.zeros((12,12))
    
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        for x in data:
            a,b = x
            prevs = ""
            
            for w,s in zip(a,b):
                self.pos[s] = self.pos.get(s, 0) + 1
                #print(w,s)
                if prevs == "":
                    self.pos_init[s] = self.pos_init.get(s, 0) + 1
                    prevs = s
                else:
                    #print(self.transition)
                    if prevs not in self.transition:  self.transition[prevs] = {}
                    if s not in self.transition[prevs]:  self.transition[prevs][s] = 1
                    else :  self.transition[prevs][s] += 1
                    prevs = s
                if w not in self.emission:
                    self.emission[w] = {'adj':1e-100,'adv':1e-100,'adp':1e-100,'conj':1e-100,'det':1e-100,'noun':1e-100,'num':1e-100,\
                                 'pron':1e-100,'prt':1e-100,'verb':1e-100,'x':1e-100,'.':1e-100}
                self.emission[w][s] += 1                    
        
        for w,val in self.emission.items():   #as word given pos tag s, so we normalize based on s
            for s in val.keys():
                val[s]= val[s]/self.pos[s]
        for s1 in self.transition:  ## modelled as new s or Si+1|Si so normalized on prev s or Si
            total = sum(self.transition[s1].values())
            for s2 in self.transition[s1]:
                self.transition[s1][s2] = self.transition[s1][s2]/total
        for s in self.pos_init.keys():  ## generate the initial probabilities
            self.pos_init_probs[s] = self.pos_init[s]/sum(self.pos_init.values())

        i = 0
        for key in self.transition.keys():
            self.mapPOS[key] = i
            i += 1
        
        self.inv_mapPOS = {value : key for key,value in self.mapPOS.items()}

        for key1 in self.transition:
            for key2 in self.transition[key1]:
                self.transitionMatrix[self.mapPOS[key1]][self.mapPOS[key2]] = self.transition[key1][key2]
        self.transitionMatrix[self.transitionMatrix == 0] = 1e-50
        #print(self.pos_init)
        #print(self.emission)
        #print(self.transition)


    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        pred = []
        for w in sentence:
            try:
                tmpdict={s:times*self.pos[s] for s,times in self.emission[w].items()}
                pred.append(max(tmpdict,key=tmpdict.get))
            except:
                pred.append(max(self.pos,key=self.pos.get))        
            
        return pred

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        #for w,s in sentence.
        #construct a transition table storing the values
        V = np.zeros((12,len(sentence)))
        i = 0
        backtrack = np.zeros(V.shape)
        for w in sentence:
            
            #initial probabilities
            if i == 0:
                for pos in self.mapPOS.keys():
                    emission = 0
                    if w not in self.emission.keys():
                        emission = 1e-5
                    else:
                        emission = self.emission[w][pos]
                    V[self.mapPOS[pos],i] = np.log(self.pos_init_probs[pos] * emission)
            else:
                #transition to next state
                for pos in self.mapPOS.keys():
                    emission = 0
                    if w not in self.emission.keys():
                        emission = -100
                    else:
                        emission = np.log(self.emission[w][pos])
                    vector = np.log(self.transitionMatrix.T[self.mapPOS[pos],:].reshape(1,-1)) + \
                                V[:,i-1].reshape(1,-1)
                    best = np.argmax(vector)
                    backtrack[self.mapPOS[pos],i] = best
                    V[self.mapPOS[pos],i] = np.max(vector) + emission
            i += 1
        #backtracking
        prev_best = np.argmax(V[:,-1])
        best_sequence = [self.inv_mapPOS[prev_best]]
        for i in range(len(sentence)-1):
            index = len(sentence) - i - 1
            prev_best = int(backtrack[prev_best,index])
            best_sequence.append(self.inv_mapPOS[prev_best])
        best_sequence.reverse()
        return best_sequence


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

