#!/usr/bin/env python3

# put your routing program here!


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 01:47:16 2018

@author: Abhilash
"""

import sys
import numpy as np

AVGSPEED = 55.0

#Parsing command line arguments...
#./route.py [start-city] [end-city] [routing-algorithm] [cost-function]
Source_city = str(sys.argv[1])
Destination_city = str(sys.argv[2])
Routing_Algorithm = str(sys.argv[3])
Cost_Funtion = str(sys.argv[4])

#File basepath
#Let's define a data structure for storing the data
#filepath for city-gps.txt
cityGpsFile = 'city-gps.txt'
cityGps_fp = open(cityGpsFile, 'r')

#filepath for road-segments.txt
roadSegmentsFile = 'road-segments.txt'
roadSegmentsFile_fp = open(roadSegmentsFile, 'r')


#A dictionary containing all the information about all the cities
#Latitude and longtitude may not be available for some of the cities.
city_map = {} 

#populating data in dictionary from city-gps.txt
for line in cityGps_fp:
    city_map[line.split()[0]] = {'lat' : line.split()[1],           #Latitude
                                 'long' : line.split()[2],          #Longitude 
                                 'connectedCities' : {},            #Dictionary of the connected cities   
                                 'path' : "",                       #path from the source to the current city
                                 'highway_path' : "",               #Path from the source to the current city in terms of the names of the highways
                                 'isVisited' : False,               #states whether the state has been visited or not, true if visited
                                 'segments' : 0,                #the cost from the source to the current city in terms of segments
                                 'distance' : 0,                    #the cost from the source to the current city in terms of distance, g(s)
                                 'heuristic' : 0,                   #hearistic cost h(s)
                                 'TotalCost' : 0,                   #Total cost, f(s) = g(s) + h(s)
                                 'time' : 0 }                       #the cost from the source to the current city in terms of time, g(s)

MaxSegmentLength = 0
MaxSpeedLimit = 0
#populating data in dictionary from road-segments.txt
for line in roadSegmentsFile_fp:
    #if any of the cities is not present in city_gps.txt then add the city to it without latitude or longitude
    if line.split()[0] not in city_map.keys():
        city_map[line.split()[0]] = {'connectedCities' : {}, \
                                     'path' : "", \
                                     'highway_path' : "", \
                                     'isVisited' : False, \
                                     'segments' : 0, \
                                     'distance' : 0, \
                                     'heuristic' : 0, \
                                     'TotalCost' : 0, \
                                     'time' : 0 }
    if line.split()[1] not in city_map.keys():
        city_map[line.split()[1]] = {'connectedCities' : {}, \
                                     'path' : "", \
                                     'highway_path' : "", \
                                     'isVisited' : False, \
                                     'segments' : 0, \
                                     'distance' : 0, \
                                     'heuristic' : 0, \
                                     'TotalCost' : 0, \
                                     'time' : 0 }
        
    if len(line.split()) == 5: #where max speed is given
        #connecting B to A. line is 'A B length speedLimit highway'
        if float(line.split()[3]) != 0:
            city_map[line.split()[0]]['connectedCities'][line.split()[1]] = \
            {'length': float(line.split()[2]), 'speedLimit' : float(line.split()[3]), 'highway' : line.split()[4]}
        else:
            city_map[line.split()[0]]['connectedCities'][line.split()[1]] = \
            {'length': float(line.split()[2]), 'speedLimit' : AVGSPEED, 'highway' : line.split()[4]}
        #connecting A to B
        if float(line.split()[3]) != 0:
            city_map[line.split()[1]]['connectedCities'][line.split()[0]] = \
            {'length': float(line.split()[2]), 'speedLimit' : float(line.split()[3]), 'highway' : line.split()[4]}
        else:
            city_map[line.split()[1]]['connectedCities'][line.split()[0]] = \
            {'length': float(line.split()[2]), 'speedLimit' : AVGSPEED, 'highway' : line.split()[4]}
        if float(line.split()[2]) > MaxSegmentLength:
            MaxSegmentLength = float(line.split()[2])
        if float(line.split()[3]) > MaxSpeedLimit:
            MaxSpeedLimit = float(line.split()[3])
    else: #where max speed is not given
        city_map[line.split()[0]]['connectedCities'][line.split()[1]] = \
        {'length': float(line.split()[2]), 'speedLimit' : AVGSPEED, 'highway' : line.split()[3]}

        city_map[line.split()[1]]['connectedCities'][line.split()[0]] = \
        {'length': float(line.split()[2]), 'speedLimit' : AVGSPEED, 'highway' : line.split()[3]}

        if float(line.split()[2]) > MaxSegmentLength:
            MaxSegmentLength = float(line.split()[2])

    
#successor function for BFS which only adds connected cities            
def successorCities(source):
    return city_map[source]['connectedCities'].keys()

#Returns the approximated eucledian distance based on lat long values
def Eucledian_Distance(source, destination):
    eucledian_distance = -1
    if("lat" and "long" in city_map[source].keys()) and ("lat" and "long" in city_map[destination].keys()):
        eucledian_distance = np.sqrt((np.square(69*(float(city_map[source]['lat']) - float(city_map[destination]['lat'])))) + \
                                     (np.square(55*(float(city_map[source]['long']) - float(city_map[destination]['long'])))))    
    return eucledian_distance*0.45

#returns true if the city has lat long attributes
def hasLatLong(city):
    if ('lat' and 'long') in city_map[city].keys():
        return True
    return False

#returns the shortest distance going deep till it first finds the city with lat long
visited = []
def goDeepfromToFindLatLong(source, destination):
    shortestdistance = sys.maxsize
    if ((("lat" and "long") in city_map[source]) and (("lat" and "long") in city_map[destination])):
        return Eucledian_Distance(source, destination)
    for city in successorCities(source):
        if city in visited:
            continue
        length = city_map[source]['connectedCities'][city]['length']
        if ("lat" or "long") not in city_map[city]:
            visited.append(source)
            dist = goDeepfromToFindLatLong(city, destination)
        else:
            dist = Eucledian_Distance(city, destination)
        distance = length + dist
        if distance < shortestdistance:
            shortestdistance = distance
    return shortestdistance

#This returns the admissible eucledian distance assuming the source and the destination have lat-long
#Assumption for return values
#    -1 no lat long for either of source or destination
#    -2 lat long buggy for source
#    -3 lat long buggy for destination
def checkAdmissibleEucledianDistance(source, destination):
    threshold = 0.8
    count_cities_validating_condition_from_source = 0
    count_cities_validating_condition_from_destination = 0
    count_successor_cities = 0
    shortestDistance = goDeepfromToFindLatLong(source, destination)
    for source_successor in successorCities(source):
        count_successor_cities += 1
        if shortestDistance < goDeepfromToFindLatLong(source_successor, destination) + goDeepfromToFindLatLong(source, source_successor):
            count_cities_validating_condition_from_source += 1
    for destination_predecessor in successorCities(destination):
        count_successor_cities += 1
        if shortestDistance < goDeepfromToFindLatLong(source, destination_predecessor) + goDeepfromToFindLatLong(destination_predecessor, destination):
            count_cities_validating_condition_from_destination += 1
    count_cities_validating_condition = count_cities_validating_condition_from_destination + count_cities_validating_condition_from_source     
    if count_cities_validating_condition > threshold*count_successor_cities:
        return shortestDistance
    elif(count_cities_validating_condition_from_destination > count_cities_validating_condition_from_source):
        return -2
    else:
        return -3

#returns the heuristic cost in terms of the cost function
def heuristicCost(city, destination):
    shortestDistance = sys.maxsize
    if hasLatLong(city) and hasLatLong(destination):
        shortestDistance = Eucledian_Distance(city, destination)
    elif not hasLatLong(city) and hasLatLong(destination):
        shortestDistance = goDeepfromToFindLatLong(city, destination)
    elif hasLatLong(city) and not hasLatLong(destination):
        shortestDistance = goDeepfromToFindLatLong(destination, city)
    else:
        found = False
        new_destination = destination
        while not found:            
            for s in successorCities(destination):
                min_d = sys.maxsize
                if hasLatLong(s):
                    shortestDistance = goDeepfromToFindLatLong(city, s) - city_map[destination]['connectedCities'][s]['length']
                    found = True
                else:
                    if city_map[destination]['connectedCities'][s]['length'] < min_d:
                        min_d = city_map[destination]['connectedCities'][s]['length']
                        new_destination = s
            destination = new_destination
    if Cost_Funtion == 'time':
        shortestDistance = shortestDistance / MaxSpeedLimit
    if Cost_Funtion == "segments":
        shortestDistance = int(shortestDistance / MaxSegmentLength)
    return shortestDistance

#Uniform search
def UniformSearch(source, destination):
    fringe = [source]
    city_map[source]['path'] = source
    city_map[source]['isVisited'] = True
    print ("Finding Path from " + str(source) + " to " +  str(destination))
    index = 0 #This is the index from which the state will be popped out from the fringe
    while(len(fringe) > 0):
        currentCity = fringe.pop(index)

        if currentCity == destination:
            print ("Path found")
            return currentCity 
                
        for next_city in successorCities(currentCity):
            incremental_cost = 0
            if Cost_Funtion == 'distance':
                incremental_cost = city_map[currentCity]['connectedCities'][next_city]['length']

            if Cost_Funtion == 'time':
                incremental_cost = city_map[currentCity]['connectedCities'][next_city]['length'] / \
                city_map[currentCity]['connectedCities'][next_city]['speedLimit']

            if Cost_Funtion == 'segments':
                incremental_cost = 1

            if city_map[next_city]['isVisited'] == True and \
               city_map[currentCity][Cost_Funtion] + incremental_cost < city_map[next_city][Cost_Funtion]:
                   
                #Add city name to the path : Path represents the path from the source to this state
                city_map[next_city]['path'] = city_map[currentCity]['path'] + ' ' + next_city
                
                #Add Highway names to the highway path
                city_map[next_city]['highway_path'] = city_map[currentCity]['highway_path'] + ' ' + \
                                    city_map[currentCity]['connectedCities'][next_city]['highway']
                #Update all the costs
                city_map[next_city]['segments'] = city_map[currentCity]['segments'] + 1 #segment cost
                city_map[next_city]['distance'] = city_map[currentCity]['distance'] + \
                                            city_map[currentCity]['connectedCities'][next_city]['length']
                city_map[next_city]['time'] = city_map[currentCity]['time'] + \
                                                (city_map[currentCity]['connectedCities'][next_city]['length']/ \
                                                 city_map[currentCity]['connectedCities'][next_city]['speedLimit'])
                
                fringe.append(next_city)
                city_map[next_city]['isVisited'] = True

            if city_map[next_city]['isVisited'] == False:
                #Add city name to the path : Path represents the path from the source to this state
                city_map[next_city]['path'] = city_map[currentCity]['path'] + ' ' + next_city
                
                #Add Highway names to the highway path
                city_map[next_city]['highway_path'] = city_map[currentCity]['highway_path'] + ' ' + \
                                    city_map[currentCity]['connectedCities'][next_city]['highway']
                #Update all the costs
                city_map[next_city]['segments'] = city_map[currentCity]['segments'] + 1 #segment cost
                city_map[next_city]['distance'] = city_map[currentCity]['distance'] + \
                                            city_map[currentCity]['connectedCities'][next_city]['length']
                city_map[next_city]['time'] = city_map[currentCity]['time'] + \
                                                (city_map[currentCity]['connectedCities'][next_city]['length']/ \
                                                 city_map[currentCity]['connectedCities'][next_city]['speedLimit'])
                
                fringe.append(next_city)
                city_map[next_city]['isVisited'] = True
                
        #update the priority index with the city having the lowest cost in distance for the uniform cost
        #Uniform Cost Search
        if Routing_Algorithm == "uniform":
            min_value = sys.maxsize
            for i in range(0, len(fringe)):
                if city_map[fringe[i]][Cost_Funtion] < min_value:
                    min_value = city_map[fringe[i]][Cost_Funtion]
                    index = i
    print ("Path not found")                
    return False


def InitializeCityMap():
    for key in city_map.keys():
        city_map[key]['isVisited'] = False
        city_map[key]['path'] = ""
        city_map[key]['highway_path'] = ""
        city_map[key]['segments'] = 0
        city_map[key]['distance'] = 0
        city_map[key]['time'] = 0
        city_map[key]['TotalCost'] = 0
        city_map[key]['heuristic'] = 0
    return

#BFS, DFS, IDS
def BFS_DFS_IDS(source, destination, depth = len(city_map)):
    fringe = [source]
    city_map[source]['path'] = source
    city_map[source]['isVisited'] = True
    index = 0 #This is the index from which the state will be popped out from the fringe
    currentDepth = 0
    while(len(fringe) > 0):
        #BFS Algorith - Default Behaviour
        if Routing_Algorithm == "bfs":
            index = 0
        
        #DFS Algorithm 
        if Routing_Algorithm == "dfs":
            index = len(fringe) - 1
        
        #DFS Algorithm 
        if Routing_Algorithm == "ids":
            index = len(fringe) - 1

        currentCity = fringe.pop(index)
        
        for next_city in successorCities(currentCity):
            if city_map[next_city]['isVisited'] == False:
                #Add city name to the path : Path represents the path from the source to this state
                city_map[next_city]['path'] = city_map[currentCity]['path'] + ' ' + next_city
                
                #Add Highway names to the highway path
                city_map[next_city]['highway_path'] = city_map[currentCity]['highway_path'] + ' ' + \
                                    city_map[currentCity]['connectedCities'][next_city]['highway']
                #Update all the costs
                city_map[next_city]['segments'] = city_map[currentCity]['segments'] + 1 #segment cost
                city_map[next_city]['distance'] = city_map[currentCity]['distance'] + \
                                            city_map[currentCity]['connectedCities'][next_city]['length']
                city_map[next_city]['time'] = city_map[currentCity]['time'] + \
                                                (city_map[currentCity]['connectedCities'][next_city]['length']/ \
                                                 city_map[currentCity]['connectedCities'][next_city]['speedLimit'])
                
                if currentCity == destination:
                    return currentCity
                if currentDepth < depth:
                    fringe.append(next_city)
                city_map[next_city]['isVisited'] = True
        currentDepth = city_map[currentCity]['segments'] + 1
    return False

#AStar 
def AStar(source, destination):
    fringe = [source]
    city_map[source]['path'] = source
    city_map[source]['isVisited'] = True
    print ("Finding Path from " + str(source) + " to " +  str(destination))
    index = 0 #This is the index from which the state will be popped out from the fringe
    while(len(fringe) > 0):

        currentCity = fringe.pop(index)

        if currentCity == destination:
            return currentCity 
                
        for next_city in successorCities(currentCity):
            incremental_cost = 0
            if Cost_Funtion == 'distance':
                incremental_cost = city_map[currentCity]['connectedCities'][next_city]['length']

            if Cost_Funtion == 'time':
                incremental_cost = city_map[currentCity]['connectedCities'][next_city]['length'] / \
                city_map[currentCity]['connectedCities'][next_city]['speedLimit']

            if Cost_Funtion == 'segments':
                incremental_cost = 1

            if city_map[next_city]['isVisited'] == True and \
               city_map[currentCity]['TotalCost'] + incremental_cost < city_map[next_city]['TotalCost']:
                   
                #Add city name to the path : Path represents the path from the source to this state
                city_map[next_city]['path'] = city_map[currentCity]['path'] + ' ' + next_city
                
                #Add Highway names to the highway path
                city_map[next_city]['highway_path'] = city_map[currentCity]['highway_path'] + ' ' + \
                                    city_map[currentCity]['connectedCities'][next_city]['highway']
                #Update all the costs
                city_map[next_city]['segments'] = city_map[currentCity]['segments'] + 1 #segment cost
                city_map[next_city]['distance'] = city_map[currentCity]['distance'] + \
                                            city_map[currentCity]['connectedCities'][next_city]['length']
                city_map[next_city]['time'] = city_map[currentCity]['time'] + \
                                                (city_map[currentCity]['connectedCities'][next_city]['length']/ \
                                                 city_map[currentCity]['connectedCities'][next_city]['speedLimit'])
                city_map[next_city]['heuristic'] = heuristicCost(next_city, destination)
                city_map[next_city]['TotalCost'] = city_map[next_city][Cost_Funtion] + city_map[next_city]['heuristic']
                
#                fringe.append(next_city)
#                city_map[next_city]['isVisited'] = True

            if city_map[next_city]['isVisited'] == False:
                #Add city name to the path : Path represents the path from the source to this state
                city_map[next_city]['path'] = city_map[currentCity]['path'] + ' ' + next_city
                
                #Add Highway names to the highway path
                city_map[next_city]['highway_path'] = city_map[currentCity]['highway_path'] + ' ' + \
                                    city_map[currentCity]['connectedCities'][next_city]['highway']
                #Update all the costs
                city_map[next_city]['segments'] = city_map[currentCity]['segments'] + 1 #segment cost
                city_map[next_city]['distance'] = city_map[currentCity]['distance'] + \
                                            city_map[currentCity]['connectedCities'][next_city]['length']
                city_map[next_city]['time'] = city_map[currentCity]['time'] + \
                                                (city_map[currentCity]['connectedCities'][next_city]['length']/ \
                                                 city_map[currentCity]['connectedCities'][next_city]['speedLimit'])
                city_map[next_city]['heuristic'] = heuristicCost(next_city, destination)
                city_map[next_city]['TotalCost'] = city_map[next_city][Cost_Funtion] + city_map[next_city]['heuristic']
                
                fringe.append(next_city)
                city_map[next_city]['isVisited'] = True
                
        #update the priority index with the city having the lowest cost in distance for the uniform cost
        #Uniform Cost Search
        if Routing_Algorithm == "astar":
            min_value = sys.maxsize
            for i in range(0, len(fringe)):
                if city_map[fringe[i]]['TotalCost'] < min_value:
                    min_value = city_map[fringe[i]]['TotalCost']
                    index = i
                    
    return False

def tourGuide():
    cities = city_map[Destination_city]['path'].split()
    highways = city_map[Destination_city]['highway_path'].split()
    if (len(cities) == 0 or len(highways) == 0):
        return
    print ("Start at " + Source_city)
    for i in range(0, len(highways)):
        print ("Take highway " + highways[i] + ", from " + cities[i] + " to " + cities[i+1])
    print()
    print("*************************************************")
    print()
    print("Total distance that will be covered in this journey : " + str(city_map[Destination_city]['distance']) + " miles")
    print("Total estimated time for this journey : " + str(city_map[Destination_city]['time']) + " hours")
    print("Total number of turns or segments in this journey : " + str(city_map[Destination_city]['segments']) + " turns/segments")
    print()
    return 0

#machine readable output
def machineOutput():
    optimal = "no"
    total_distance = city_map[Destination_city]['distance']
    total_time = city_map[Destination_city]['time']
    traveresed_cities = city_map[Destination_city]['path']
    if Routing_Algorithm == "uniform":
        optimal = "yes"

    if Routing_Algorithm == "ids" and Cost_Funtion == "segments":
        optimal = "yes"
    
    if Routing_Algorithm == "bfs" and Cost_Funtion == "segments":
        optimal = "yes"
    
    return optimal + " " + str(total_distance) + " " + str(total_time) + " " + traveresed_cities

#Solve search problem
def SolveSearch():
    
    if Source_city not in city_map.keys() or Destination_city not in city_map.keys():
        print("Please enter valid source and destination city names given in the text file")
        return
    if Cost_Funtion not in ["distance", "segments", "time"]:
        print ("Please enter one of the cost funtions - distance, time, segments")
        return
    
    if Routing_Algorithm == "astar":
        AStar(Source_city, Destination_city)
    elif Routing_Algorithm == "uniform":
        UniformSearch(Source_city, Destination_city)
    elif Routing_Algorithm == "ids":
        i = 0
        found = False
        destinationCity = False
        while i < len(city_map) and not found:
            InitializeCityMap()
            destinationCity = BFS_DFS_IDS(Source_city, Destination_city, i)
            if destinationCity:
                found = True
            i += 1 
    elif Routing_Algorithm == "dfs" or Routing_Algorithm == "bfs":
        BFS_DFS_IDS(Source_city, Destination_city)
    else:
        print("Please enter a valid name of search algorithm")
    
    tourGuide()
    print(machineOutput())
    return

SolveSearch()