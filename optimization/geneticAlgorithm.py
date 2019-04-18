"""
Genetic Algorithm used to minimize positive functions.
"""

import numpy as np
import math
import random

class GeneticAlgorithm:
    def __init__(self, minFunction, inputSize, lowerBound, upperBound,
                 maxIteractions, populationSize, eliteNum,
                 threshold = np.NINF, selectionMethod = None, mutationMethod = None, crossoverMethod = None,
                 chromosomeMutationRate = 0.2, geneMutationRate = 0.01, tournamentSize = 5):

        # Function to be optimized.
        self.geneSize = inputSize
        self.minFunction = minFunction
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        # Parameters of GeneticAlgorithm.
        self.maxIteractions = maxIteractions
        self.populationSize = populationSize
        self.eliteNum = eliteNum
        self.threshold = threshold

        # Methods of GeneticAlgorithm.
        self.selectCouples = selectionMethod if selectionMethod else GeneticAlgorithm.tournamentSelect
        self.mutation = mutationMethod if mutationMethod else GeneticAlgorithm.geneMutation
        self.crossover = crossoverMethod if crossoverMethod else GeneticAlgorithm.singlePointCrossover

        # Parameters of methods.
        self.crossoverNum = self.populationSize - self.eliteNum
        self.chromosomeMutationRate = chromosomeMutationRate
        self.geneMutationRate = geneMutationRate
        self.tournamentSize = tournamentSize


    """
    Methods of selection of couples (one of them must be chosen).
    """
    # Roulette Wheel selection.
    def wheelSelect(self):
        roulette = self.fitValues[-1] - self.fitValues
        roulette /= np.add.reduce(roulette)
        return np.random.choice(self.populationSize, size = (2, self.crossoverNum), p = roulette)


    # Tournament selection.
    def tournamentSelect(self):
        tournament = np.random.randint(0, self.populationSize, (self.tournamentSize, 2*self.crossoverNum))
        winners = np.min(tournament, axis = 0)
        return winners.reshape(2, self.crossoverNum)


    # Stochastic selection
    def stochasticSelect(self):
        fit = self.fitValues[-1] - self.fitValues
        cumsum = np.cumsum(fit)
        distance = cumsum[-1] / (2*self.crossoverNum)
        parents = np.searchsorted(cumsum, np.arange(distance * random.random(), cumsum[-1], distance))
        return parents.reshape(2, self.crossoverNum)


    """
    Methods of mutation (one of them must be chosen).
    """
    # This function has a chance of choosing each chromosome.
    # The chromosome chosen will have a random gene changed to a random value.
    def chromosomeMutation(self, population):
        elements = np.nonzero(np.random.rand(self.crossoverNum) < self.chromosomeMutationRate)[0]
        positions = np.random.randint(0, self.geneSize, elements.size)
        population[elements, positions] = np.random.uniform(self.lowerBound[positions], self.upperBound[positions], elements.size)
        return population


    # This function has a chance of choosing each gene.
    # Every gene chosen will be changed to a random value.
    def geneMutation(self, population):
        mutation = np.random.rand(*population.shape) < self.geneMutationRate
        population[mutation] = np.random.uniform(self.lowerBound[None, :].repeat(self.crossoverNum, 0)[mutation],
                                                 self.upperBound[None, :].repeat(self.crossoverNum, 0)[mutation],
                                                 np.count_nonzero(mutation))
        return population


    """
    Methods linked to the crossover.
    """
    # Creates a boolean mask.
    def booleanMask(self, height, width):
        num = height * width
        numBytes = -(-num // 8)
        seqBytes = np.frombuffer(np.random.bytes(numBytes), np.uint8)
        return np.unpackbits(seqBytes)[:num].reshape(height, width)


    # Makes the crossover between two chromosomes randomly choosing the source of each gene.
    def uniformCrossover(self, couples):
        truth = self.booleanMask(couples.size // 2, self.geneSize)
        newPopulation = np.where(truth, self.population[couples[0]], self.population[couples[1]])
        return newPopulation


    # Makes the crossover between two chromosomes using genes from one parent before a random point and
    # from the other parent after that point.
    def singlePointCrossover(self, couples):
        truth = np.arange(self.geneSize) < np.random.randint(0, self.geneSize + 1, couples.size // 2)[:, None]
        newPopulation = np.where(truth, self.population[couples[0]], self.population[couples[1]])
        return newPopulation


    # Makes the crossover between two chromosomes using genes from one parent between two random points and
    # from the other parent outside the interval defined by the points.
    def twoPointCrossover(self, couples):
        grid = np.arange(self.geneSize)
        rand = np.sort(np.random.randint(0, self.geneSize + 1, (2, couples.size // 2, 1)))
        rand[0] -= 1
        truth = (grid >= rand[0]) & (grid < rand[1])
        newPopulation = np.where(truth, self.population[couples[0]], self.population[couples[1]])
        return newPopulation


    # Don't make any crossover.
    def noCrossover(self, couples):
        return self.population[couples[0]]


    """
    Methods related to the population and the generations.
    """
    def sortPopulation(self):
        argSort = self.fitValues.argsort()
        self.fitValues = self.fitValues[argSort]
        self.population = self.population[argSort]


    def firstPopulation(self):
        self.population = np.random.uniform(self.lowerBound, self.upperBound, (self.populationSize, self.geneSize))
        self.fitValues = self.minFunction(self.population.T)
        self.sortPopulation()


    def nextGeneration(self):
        offspring = self.mutation(self, self.crossover(self, self.selectCouples(self)))
        self.population[self.eliteNum:] = offspring
        self.fitValues[self.eliteNum:] = self.minFunction(offspring.T)
        self.sortPopulation()

   
    """
    Iterates over the generations.
    """
    def run(self):
        self.firstPopulation()

        for _ in range(self.maxIteractions):
            if self.fitValues[0] < self.threshold:
                break

            self.nextGeneration()

        # Returns the chromosome with best fit on the population, since it's ordered.
        return self.population[0]
