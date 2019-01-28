#!/usr/bin/env python2

from datetime import datetime 
import sys

minimum = 0
got_first_minimum = False
n = int(sys.argv[4])
m = int(sys.argv[3])
k = int(sys.argv[2])

class User:
    def __init__(self, uname, exp_team_size, want, not_want):
        self.uname = uname
        self.exp_team_size = exp_team_size
        self.want = want
        self.not_want = not_want

class Node:
    def __init__(self, team, cost, selected_teams, remaining_students):
        self.team = team
        self.cost = cost
        self.selected_teams = selected_teams
        self.remaining_students = remaining_students
    
def cost_function(s):
    cost = 0
    for t in s:
        if pool[t].exp_team_size != 0 and pool[t].exp_team_size != len(s):
            cost += 1
        for r in pool[t].want:
            if r not in s:
                cost += n;
        for r in pool[t].not_want:
            if r in s:
                cost += m;
    cost += k;
    return cost

def processInput(filepath):
    file = open(filepath)
    a = file.readlines()
    pool = {}
    for i in a:
        i = i.strip()
        l = i.split(" ")
        [uname, size, want, unwant] = l
        want_id = [p for p in want.split(",") if want is not "_"]
        unwant_id = [q for q in unwant.split(",") if unwant is not "_"]
        x = User(uname, int(size), want_id, unwant_id)
        pool[x.uname] = x
    return pool

def extractFullConsent(pool):
    pool1 = []
    pool1_cost = 0
    for key1 in pool:
        bool = []
        for key2 in pool[key1].want:
            bool.append(set([key1] + pool[key1].want) == set([key2] + pool[key2].want))
        if len(bool) > 0 and False not in bool:
            s1 = set([key1] + pool[key1].want)
            if s1 not in pool1:
                pool1.append(s1)
                pool1_cost += cost_function(tuple(s1))
    for w in pool1: 
        for r in w:
            del(pool[r])
    return pool, pool1, pool1_cost

def generateSampleSpace(pool):
    sample_state = {}
    for key1 in pool:
        sample_state[(key1)] = cost_function([key1])
        s1 = set(pool) - set({key1:pool[key1]})
        for key2 in s1:
            key = sorted((key1, key2))
            key = tuple(key)
            if key not in sample_state:
                sample_state[key] = cost_function([key1, key2])
                s2 = set(s1) - set({key2:pool[key2]})
                for key3 in s2:
                    key = sorted((key1, key2, key3))
                    key = tuple(key)
                    if key not in sample_state:
                        sample_state[key] = cost_function([key1, key2, key3])
    return sample_state

def generateGroupsSizewise(sorted_sample_state):
    group_size_1 = []
    group_size_2 = []
    group_size_3 = []

    for group in sorted_sample_state:
        if type(group) == str:
            group_size_1.append(group)
        elif len(group) == 2:
            group_size_2.append(group)
        else:
            group_size_3.append(group)
    return group_size_1, group_size_2, group_size_3

def generateSuccessorOfSize_1(node):
    for s1 in group_size_1:
        if got_first_minimum and node.cost + sample_state[s1] > minimum:
            break
        if s1 in node.remaining_students:
            selected_teams = [sel_team for sel_team in node.selected_teams]
            selected_teams.append(s1)
            cost = node.cost + sample_state[s1]
            rem_st = [st for st in node.remaining_students]
            remaining_students = rem_st.remove(s1)
            new_node = Node(s1, cost, selected_teams, remaining_students)
            return new_node
        
def generateSuccessorOfSize_2(node):
    for s1 in group_size_2:
        if got_first_minimum and node.cost + sample_state[s1] > minimum:
            break
        if len(set(s1) - set(node.remaining_students)) == 0:
            selected_teams = [sel_team for sel_team in node.selected_teams]
            selected_teams.append(s1)
            cost = node.cost + sample_state[s1]
            rem_st = [st for st in node.remaining_students]
            remaining_students = tuple(set(rem_st) - set(s1))
            new_node = Node(s1, cost, selected_teams, remaining_students)
            return new_node
        
def generateSuccessorOfSize_3(node):
    for s1 in group_size_3:
        if got_first_minimum and node.cost + sample_state[s1] > minimum:
            break
        if len(set(s1) - set(node.remaining_students)) == 0:
            selected_teams = [sel_team for sel_team in node.selected_teams]
            selected_teams.append(s1)
            cost = node.cost + sample_state[s1]
            rem_st = [st for st in node.remaining_students]
            remaining_students = tuple(set(rem_st) - set(s1))
            new_node = Node(s1, cost, selected_teams, remaining_students)
            return new_node

def successors(node):
    successor = []
    succ1 = generateSuccessorOfSize_1(node)
    if succ1 != None:
        successor.append(succ1)
    
    succ2 = generateSuccessorOfSize_2(node)
    if succ2 != None:
        successor.append(succ2)
        
    succ3 = generateSuccessorOfSize_3(node)
    if succ3 != None:
        successor.append(succ3)
    
    return successor

def generateGroups(sample_state):
    fringe = []
    min_teams = []
    minimum = 0
    got_first_minimum = False

    for team in sample_state:
        cost = sample_state[team]
        selected_teams = []
        selected_teams.append(team)
        
        team_list = []
        if type(team) == str:
            team_list.append(team)
        else:
            team_list = team

        remaining_students = tuple(set(pool.keys()) - set([student for student in team_list]))

        node = Node(team_list, cost, selected_teams, remaining_students)
        fringe.append(node)

        while len(fringe) > 0 :
            N = fringe.pop()
            lenRemStd = 0
            if N.remaining_students != None:
                lenRemStd = len(N.remaining_students)

            if lenRemStd == 0 and got_first_minimum == False:
                got_first_minimum = True
                minimum = N.cost
                min_teams = N.selected_teams
            
            if got_first_minimum and N.cost > minimum:
                break
            
            if lenRemStd != 0 :
                for successor in successors(N): 
                    fringe.append(successor)

            flag = not got_first_minimum 
            if flag:
                minimum = N.cost
                min_teams = N.selected_teams
        
            if got_first_minimum and lenRemStd == 0 and N.cost < minimum:
                minimum = N.cost
                min_teams = N.selected_teams 
        
        if (datetime.now() - start).seconds > len(pool)/3:        
            break
    return minimum, min_teams

start = datetime.now()

filePath = sys.argv[1]
#pool = processInput("C:\Users\Darshan\Desktop\sample_data_for_q3.txt")
#pool = processInput("C:\Users\Darshan\Desktop\data_for_q3.txt")
#pool = processInput("C:\Users\Darshan\Desktop\input_large_200.txt")

pool = processInput(filePath)
pool, pool1, pool1_cost = extractFullConsent(pool)
pool1 = [tuple(grp) for grp in pool1]

sample_state =  generateSampleSpace(pool)
sorted_sample_state = {}
for c1 in sorted(sample_state.iteritems(), key=lambda (k,v): (v,k)):
    sorted_sample_state[c1[0]] = c1[1]

sorted_sample_state_list = sorted(sample_state, key=sample_state.__getitem__)
group_size_1, group_size_2, group_size_3 = generateGroupsSizewise(sorted_sample_state_list)
minimum, min_teams =  generateGroups(sorted_sample_state)
min_teams += pool1       
end = datetime.now()

for final_team in min_teams:
    strg = ""
    if type(final_team) == str:
        strg = final_team
    else:
        for mem in final_team:
             strg = strg + mem + " " 
    print strg

print str(minimum + pool1_cost)
