import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it
import random as rd
import time
import csv


def calulateDistance(data,combination):
    #calculates the distance btween cities and stores it
    finalDistance = 0
    for i in range(len(combination) - 1):
        fromCity = data[0].index(combination[i])
        toCity = data[0].index(combination[i + 1])
        distances = data[1:]
        distance = float(distances[fromCity][toCity])
        finalDistance += distance

    lastCity = data[0].index(combination[-1])
    firstCity = data[0].index(combination[0])
    lastDistance = float(data[1:][lastCity][firstCity])
    finalDistance += lastDistance
    return finalDistance


def calculateFitness(fitnessOfTour,population,distanceOfTour):
    #calculates fitness and stores it
    for i in range(len(population)):
        fitness = np.sum(distanceOfTour)/distanceOfTour[i]
        fitnessOfTour.append(fitness)
    return fitnessOfTour

def makeAPopulation(data,tourOfCities,populationSize):
    #creating a population by using the random.shuffle()
    distanceOfTour = []
    population = []
    fitnessOfTour = []
    for i in range(populationSize):
        tour = tourOfCities.copy()
        #rd.shuffle(tour)

        population.append(tour)
        distanceOfTour.append(calulateDistance(data,tour))

    fitnessOfTour = calculateFitness(fitnessOfTour,population,distanceOfTour)

    return population,distanceOfTour,fitnessOfTour


#Selection of parents
def parentSelection(population, fitnessOfTour):
    #defining the selectedParent as None
    selectedParent = None
    indeks = rd.randint(0, len(population) - 1)
    sumFitness = np.sum(fitnessOfTour)

    #loops until we find a parent
    while selectedParent == None:
        if indeks == len(population):
            indeks = 0
        #selecting the parent with highest fitness
        probability = fitnessOfTour[indeks]/sumFitness
        if probability > rd.uniform(0,1):
            selectedParent = population[indeks]
        indeks += 1

    return selectedParent

#Selection of survivor:
def survivorSelection(population, populationSize, fitnessOfTour,distanceOfTour):

    fullPopulation = []
    #adding up population,fitness and distance in one list in order to make it easier for sorting
    for i in range(len(population)):
        fullPopulation.append([population[i],distanceOfTour[i],fitnessOfTour[i]])
    #sorting and revoming those with the worst fitness
    fullPopulation.sort(key=lambda x: x[2], reverse=True)
    while len(fullPopulation) > populationSize:
        fullPopulation.pop()

    population = [row[0] for row in fullPopulation]
    distanceOfTour = [row[1] for row in fullPopulation]
    fitnessOfTour = [row[2] for row in fullPopulation]

    return population,distanceOfTour,fitnessOfTour


def pmx(a, b, start, stop):
    child = [None]*len(a)
    child[start:stop] = a[start:stop]
    for ind, x in enumerate(b[start:stop]):
        ind += start
        if x not in child:
            while child[ind] != None:
                ind = b.index(a[ind])
            child[ind] = x

    for ind, x in enumerate(child):
        if x == None:
            child[ind] = b[ind]

    return child

def pmxPair(a,b):

    half = len(a) // 2
    start = np.random.randint(0, len(a)-half)
    stop = start+half
    return pmx(a, b, start, stop), pmx(b, a, start, stop)

def swapMutation(child):
    cities = np.random.choice(len(child), 2, replace=False)
    child[cities[0]], child[cities[1]] = child[cities[1]], child[cities[0]]
    return child


#this funciton generate parents and childs by mutation and crossover
def nextGeneration(population,fitnessOfTour,data,distanceOfTour):
    populationSize = len(population)
    listOffspring = []
    distOffspring = []
    fitOffspring = []

    for i in range(len(population)//2):
        #calling on and creating parents
        parent1 = parentSelection(population, fitnessOfTour)
        parent2 = parentSelection(population, fitnessOfTour)

        #mutation and crossover
        child1, child2 = pmxPair(parent1,parent2)
        child1 = swapMutation(child1)
        child2 = swapMutation(child2)

        listOffspring.append(child1)
        distOffspring.append(calulateDistance(data,child1))
        fitOffspring.append(0)

        listOffspring.append(child2)
        distOffspring.append(calulateDistance(data,child2))
        fitOffspring.append(0)

    for i in range(len(listOffspring)):
        fitOffspring[i] = np.sum(distOffspring)/distOffspring[i]

    population = population + listOffspring
    distanceOfTour = distanceOfTour + distOffspring
    fitnessOfTour = fitnessOfTour + fitOffspring

    return survivorSelection(population, populationSize, fitnessOfTour,distanceOfTour)



def evaluation(population,fitnessOfTour,data,distanceOfTour):
    bestDist = distanceOfTour
    bestRout = population
    bestFitness = fitnessOfTour

    generation = 0
    counter = 0

    #using while loop to find the best solution by running it many times, if it runs without
    #improving the programe will terminate
    while counter < 100:
        population,distanceOfTour,fitnessOfTour = nextGeneration(population,fitnessOfTour,data,distanceOfTour)
        generation += 1

        if distanceOfTour[0] < bestDist[0]:
            bestDist = distanceOfTour
            bestRout = population
            bestFitness = fitnessOfTour
            counter = 0
        else:
            counter += 1
    return bestRout,bestDist,bestFitness,generation

def runningGA(populationSize,numberOfCities,data):
    t1 = time.time()
    tourOfCities = data[0][:numberOfCities]
    population,distanceOfTour,fitnessOfTour = makeAPopulation(data,tourOfCities,populationSize)

    bestSolution = []
    runs = []
    y = []
    for i in range(20):
        bestRout,bestDist,bestFitness,generation = evaluation(population,fitnessOfTour,data,distanceOfTour)
        y = bestDist
        bestSolution.append([bestRout[0],bestDist[0],bestFitness[0]])
        runs.append(generation)

    t2 = time.time()

    bestSolution.sort(key=lambda x: x[1], reverse=False)
    avrageFitness = [x[2] for x in bestSolution]
    gen = np.linspace(0,max(runs),len(y))
    plt.plot(gen,y,label = f"popSize = {populationSize}")

    print(f'GA for {numberOfCities} cities with populationSize ={populationSize}:')
    print(f'Shortest distance: {bestSolution[0][1]:.3f}')
    print(f'Mean distance: {np.mean([x[1] for x in bestSolution]):.3f}')
    print(f'Standard deviation: {np.std([x[1] for x in bestSolution]):.3f}')
    print(f'Worst distance: {bestSolution[-1][1]:.3f}')
    finalTime = t2-t1
    print(f"Time: {finalTime:.3f} s")
    print(f'{np.mean(runs)}')
    print("___________________________________________")

def printing(popSizes,numCities):
    data = list(csv.reader(open("european_cities.csv", "r"), delimiter=";"))
    for p in popSizes:
        runningGA(p,numCities,data)

    plt.suptitle(f"{numCities} cities")
    plt.xlabel("Generations")
    plt.ylabel("Avrage fitness")
    plt.legend()
    plt.show()

printing([20,40,80],10)
#printing([20,40,80],24)
