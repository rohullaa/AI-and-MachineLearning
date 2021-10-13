import numpy as np
import pandas as pd
import itertools as it
import random as rd
import time
import csv

data = pd.read_csv("european_cities.csv",sep=';').values.tolist()

def totalDistance(cityData,solution):
    #finds the totaldistance for a given solution.
    distance = 0
    for i in range(len(solution)):
        distance += cityData[solution[i-1]][solution[i]]
    return distance

def shortestPath(cityData,cityNum):
    bestDistance = 100000000
    bestSequence = None
    #using itertool to make permutations
    for perm in it.permutations(cityNum):
        #calculating the totalDistance for each permutations
        currentDistance = totalDistance(cityData,perm)
        #checking if the currentDistance is better than best, if thats
        #the case then it changes
        if currentDistance < bestDistance:
            bestDistance = currentDistance
            bestSequence = perm
    return bestDistance,perm

if __name__ == '__main__':
    cityNum = range(10)
    t1 = time.time()
    best_distance,best_sequece = shortestPath(data,cityNum)
    t2 = time.time()
    finalTime = t2-t1
    print(f"Best solution: {best_distance}")
    print(f"The actual sequence of the citites: {best_sequece}")
    print(f"Best time: {finalTime:.2f} s")
