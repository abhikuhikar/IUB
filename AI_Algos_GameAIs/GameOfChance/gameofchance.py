#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:23:03 2018

@author: Abhilash
"""

#The output of this file will be the names of the rolls that should be rolled again
#The rolls are represented as A, B, C. If A and B are to be rolled again the output will be AB and so on..
import sys

initial_roll = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]

def replaceA(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        if roll[1] == roll[2] == i:
            reward = 25
        else:
            reward = roll[1] + roll[2] + i
        expectedvalue = expectedvalue + reward
    
    return expectedvalue/6

def replaceB(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        if roll[0] == roll[2] == i:
            reward = 25
        else:
            reward = roll[0] + roll[2] + i
        expectedvalue = expectedvalue + reward
    
    return expectedvalue/6

def replaceC(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        if roll[0] == roll[1] == i:
            reward = 25
        else:
            reward = roll[0] + roll[1] + i
        expectedvalue = expectedvalue + reward
    
    return expectedvalue/6

def replaceAB(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        for j in [1, 2, 3, 4, 5, 6]:
            if roll[2] == j == i:
                reward = 25
            else:
                reward = roll[2] + i + j
            expectedvalue = expectedvalue + reward
            
    return expectedvalue/36

def replaceBC(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        for j in [1, 2, 3, 4, 5, 6]:
            if roll[0] == j == i:
                reward = 25
            else:
                reward = roll[0] + i + j
            expectedvalue = expectedvalue + reward
            
    return expectedvalue/36

def replaceAC(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        for j in [1, 2, 3, 4, 5, 6]:
            if roll[1] == j == i:
                reward = 25
            else:
                reward = roll[1] + i + j
            expectedvalue = expectedvalue + reward
            
    return expectedvalue/36

def replaceABC(roll):
    reward = 0
    expectedvalue = 0
    for i in [1, 2, 3, 4, 5, 6]:
        for j in [1, 2, 3, 4, 5, 6]:
            for k in [1, 2, 3, 4, 5, 6]:
                if i == j == k:
                    reward = 25
                else:
                    reward = i + j + k
                expectedvalue = expectedvalue + reward
            
    return expectedvalue/216

chance_table = {}
def maxNode(roll):
    if(int(sys.argv[1]) > 6 or int(sys.argv[2]) > 6 or int(sys.argv[3]) > 6):
        return ("Please enter valid dice numbers between 1 to 6")
    reward = 0
    if roll.count(roll[0]) == 3:
        reward = 25
    else:
        reward = sum(roll)
    chance_table["No Roll"] = reward
    chance_table["A"] = replaceA(initial_roll)
    chance_table["B"] = replaceB(initial_roll)
    chance_table["C"] = replaceC(initial_roll)
    chance_table["AB"] = replaceAB(initial_roll)
    chance_table["AC"] = replaceAC(initial_roll)
    chance_table["BC"] = replaceBC(initial_roll)
    chance_table["ABC"] = replaceABC(initial_roll)
    
    
    return max(chance_table, key=chance_table.get)
    
print (maxNode(initial_roll))