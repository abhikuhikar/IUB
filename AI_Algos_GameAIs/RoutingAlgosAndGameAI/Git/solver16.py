#!/usr/bin/env python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
#
#from queue import PriorityQueue
#from random import randrange, sample
import sys
#import ast
#import string

# shift a specified row left (1) or right (-1)
def shift_row(state, row, dir):
    change_row = state[(row*4):(row*4+4)]
    return ( state[:(row*4)] + change_row[-dir:] + change_row[:-dir] + state[(row*4+4):],\
                   ("L" if dir == -1 else "R") + str(row+1)) 

# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]
    return (s, ("U" if dir == -1 else "D") + str(col+1) )

# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print ('%3d %3d %3d %3d' % (row[j:(j+4)]))

#Circular Manhattan
def heuristic(state):
    h = 0
    for i in range(len(state)):
        row_i,col_i = int(i/4),int(i%4)
        row_t,col_t = int((int(state[i])-1)/4),int((int(state[i])-1)%4)
        h = h + min(abs(row_i-row_t),4-abs(row_i-row_t)) \
            + min(abs(col_i-col_t),4-abs(col_i-col_t))
    return h/4

#manhattan distance
def heuristic2(state):
    h = 0
    for i in range(0,16):
            row_i,col_i = state[i]//4,state[i]%4
            row_t,col_t = (state[i]-1)//4,(state[i]-1)%4
            h = h + abs(row_i-row_t) + abs(col_i-col_t)
    return h/4
    
# reverse permutation
def heuristic3(state):
    h = 0
    for i in range(0,16):
        temp = [j for j in state[i:] if j<state[i]]
        h += len(temp)
    return h

# return a list of possible successor states from current state
def successors(state):
    state_s = state["state"]
    row = row_in_order(state_s)
    col = col_in_order(state_s)
    d_succ= {}
    for i in range(0,4):
        for d in (1,-1):
            if (i not in row):
                next_state = shift_row(state_s, i, d)            
                c_state = next_state[0]
                c_path = next_state[1]
                d_succ[str(c_state)]={'state':c_state,\
                                          'g(s)':state["g(s)"]+1,\
                                          'cost':heuristic(c_state)+state["g(s)"]+1,\
                                          'path':c_path}
            if (i not in col):
                next_state = shift_col(state_s, i, d) 
                c_state = next_state[0]
                c_path = next_state[1]
                d_succ[str(c_state)]={'state':c_state,\
                                          'g(s)':state["g(s)"]+1,\
                                          'cost':heuristic(c_state)+state["g(s)"]+1,\
                                          'path':c_path}
    return d_succ

# successor function to calculate states from goal node
def successors2(state):
    state_s = state["state"]
    d_succ= {}
    for i in range(0,4):
        for d in (1,-1):
            
                next_state = shift_row(state_s, i, d)            
                c_state = next_state[0]
                c_path = next_state[1]
                d_succ[str(c_state)]={'state':c_state,\
                                          'g(s)':state["g(s)"]+1,\
                                          'cost':heuristic(c_state)+state["g(s)"]+1,\
                                          'path':c_path}
            
                next_state = shift_col(state_s, i, d) 
                c_state = next_state[0]
                c_path = next_state[1]
                d_succ[str(c_state)]={'state':c_state,\
                                          'g(s)':state["g(s)"]+1,\
                                          'cost':heuristic(c_state)+state["g(s)"]+1,\
                                          'path':c_path}
    return d_succ

# Checking rows in order
def row_in_order(state):
    row = []
    if (state[:4] == [1,2,3,4]):row.append(0)
    if (state[4:8] == [5,6,7,8]):row.append(1)
    if (state[8:12] == [9,10,11,12]):row.append(2)
    if (state[12:] == [13,14,15,16]):row.append(3)
    return row

#Checking colums in order
def col_in_order(state):
    col = []
    if (state[::4] == [1,5,9,13]):col.append(0)
    if (state[1::4] == [2,6,10,14]):col.append(1)
    if (state[2::4] == [3,7,11,15]):col.append(2)
    if (state[3::4] == [4,8,12,16]):col.append(3)
    return col

# check if we've reached the goal
def is_goal(state):
    return sorted(state) == list(state)

#goal condition to be run with precalculation of states in depth 4 from goal state
def is_goal2(state,list_of_states):
    if str(state) in list_of_states.keys():
        return True
    else:
        return False
   
def path_reverse(s):
    s1 = ""
    for i in range(len(s)):
        if(s[i]=="D"): s1 = s1+"U"
        elif(s[i]=="U"):s1 = s1+"D"
        elif(s[i]=="R"):s1 = s1+"L"
        elif(s[i]=="L"):s1 = s1+"R"
        else: s1 = s1+s[i]
    return " ".join(s1.split(" ")[::-1])

def states_till_depth4():
    start_state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    fringe = []
    depth = 0
    fringe = [{'state':start_state,\
               'path':"",\
               'cost':heuristic(start_state),\
               'g(s)':0,\
               'depth':depth}]
    k=0
    while(len(fringe)<16*16*16*16*2):
        curr_state = fringe[k]
        route_so_far = fringe[k]['path']
        succ = successors2(curr_state)
        for i in succ.keys():
            fringe.append({'state':succ[i]["state"],\
                           'path':route_so_far + " " +succ[i]["path"],\
                           'g(s)':succ[i]["g(s)"],\
                           'cost':succ[i]["cost"],\
                           'depth':len(route_so_far + " " +succ[i]["path"])//3 + 1})  
        k=k+1
    level = {}
    for i in fringe:
            if str(i['state']) not in level.keys():
                level[str(i['state'])] = {'state':i["state"],\
                                          'path':i["path"],\
                                          'cost':i["cost"],\
                                          'depth':i["depth"]}
            else:
                if i['cost'] < level[str(i['state'])]["cost"]:
                    level[str(i['state'])]["cost"] = i['cost']
                    level[str(i['state'])]["path"] = i['path']
                    
    return level
            

#The solver! - using BFS right now
def solve(initial_board):
    initial_board = list(initial_board)
    #states = states_till_depth4()
    if(is_goal(initial_board)):
        return "",initial_board
    
    #if(is_goal2(initial_board,states)):
     #   return path_reverse(states[str(initial_board)]["path"])
    
    state_to_pull = 0
    fringe = {str(initial_board):{'state':initial_board,\
                                  'g(s)':0,\
                                  'cost':heuristic(initial_board),\
                                  'path':""}}
    
    
    route_so_far=""
    while (len(fringe) > 0):
        state_to_pull = min(list(fringe)[::-1], key=lambda x:fringe[x]["cost"])
        
        curr_state = fringe.pop(state_to_pull)
        route_so_far = curr_state["path"]
        
        if(is_goal(curr_state["state"])):
            return curr_state["path"]
        
       # if(is_goal2(curr_state["state"],states)):
        #    return route_so_far+" "+path_reverse(states[str(curr_state["state"])]["path"])
        
        succ = successors(curr_state)
        for state in succ.keys():
            if (state in fringe.keys() and succ[state]["cost"]<fringe[state]["cost"]):
                fringe[state]["cost"] = succ[state]["cost"]
            elif (state in fringe.keys() and succ[state]["cost"]>=fringe[state]["cost"]):
                continue    
            if state not in fringe.keys():
                fringe[state] = {'state':succ[state]["state"],\
                                     'g(s)':succ[state]["g(s)"],\
                                     'cost':succ[state]["cost"],\
                                     'path':route_so_far+" "+succ[state]["path"]}
    return False

# test cases
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [ int(i) for i in line.split() ]
print(start_state)

if len(start_state) != 16:
    print ("Error: couldn't parse start state file")

print ("Start state: ")
print_board(tuple(start_state))

print ("Solving...")
route = solve(tuple(start_state))

if route:
    print ("Solution found in " + str(len(route)//3) +" moves:" + "\n" + route)
else: 
    print("No Solution found....")
