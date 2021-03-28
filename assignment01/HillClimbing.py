import numpy as np
import pandas as pd
import random as rd
import time
import csv
from ExhaustiveSearch import totalDistance

data = pd.read_csv("european_cities.csv",sep=';').values.tolist()

def randomSolution(cityData):
    #this function selects a random city
    cities = list(range(len(cityData)))
    solution = []

    for i in range(len(cityData)):
        #selecting a city randomly using randint
        #adding it in the list of solution
        #making sure to not select it again by removing it from cities
        randomCity = cities[rd.randint(0,len(cities)-1)]
        solution.append(randomCity)
        cities.remove(randomCity)

    return solution

def getNeighbors(solution):
    #this function returns all neighbors of the given solution
    #by recombination/swaping elements of the solution listl
    AllNeighbors = []
    for i in range(len(solution)):
        for j in range(i+1,len(solution)):
            neighbor = solution.copy()
            neighbor[i] = solution[j]
            neighbor[j] = solution[i]
            AllNeighbors.append(neighbor)
    return AllNeighbors


def getBestNeighbor(cityData,AllNeighbors):
    #initialising the current best distances and neighbors
    #this function finds the best neighbor (meaning neighbor with shortest distance)
    #by searching all of the neighbors
    bestDistance = totalDistance(cityData, AllNeighbors[0])
    bestNeighbor = AllNeighbors[0]
    for neighbor in AllNeighbors:
        currentDistance = totalDistance(cityData,neighbor)
        #checks if the currentDistance is better than the best, if so
        #it changes the best
        if currentDistance < bestDistance:
            bestDistance = currentDistance
            bestNeighbor = neighbor
    return bestNeighbor,bestDistance


def hillClimbing(cityData):
    #calling on the functions above to get the variables
    currentSolution = randomSolution(cityData)
    currentDistance = totalDistance(cityData,currentSolution)
    AllNeighbors = getNeighbors(currentSolution)
    bestNeighbor,bestDistance = getBestNeighbor(cityData,AllNeighbors)
    #checks if the currentDistance is better than the best, if so
    #it changes the best
    while bestDistance < currentDistance:
        currentSolution = bestNeighbor
        currentDistance = bestDistance
        AllNeighbors = getNeighbors(currentSolution)
        bestNeighbor,bestDistance = getBestNeighbor(cityData,AllNeighbors)

    return currentSolution,currentDistance

if __name__ == '__main__':
    amountRuns = 20
    listFirstTen = []
    listAllCities = []
    for i in range(amountRuns):
        distanceForTen = hillClimbing(data[0:10])[1]
        listFirstTen.append(distanceForTen)

        distanceAll = hillClimbing(data)[1]
        listAllCities.append(distanceAll)

    def printing(string,liste):
        print(f"Hill climbing: 20 runs of the {string}")
        print(f"Best length of the tour: {min(liste):.3f}")
        print(f"Worst length of the tour: {max(liste):.3f}")
        print(f"Mean length of the tour: {np.mean(liste):.3f}")
        print(f"Standard deviation of the runs: {np.std(liste):.3f}")
        print("___________________________________________")

    printing("first ten cities",listFirstTen)
    printing("all cities",listAllCities)
