"""
Genetic Algorithm used to minimize a positive function.
"""

import numpy as np
import math
import random

class geneticAlgorithm:
    def __init__(self, function, nArgs, lowerBound, upperBound, maxIteractions, populationSize, threshold, eliteNum,
                 selectionMethod, mutationMethod, crossoverMethod,
                 chromosomeMutationRate = 0.2, geneMutationRate = 0.01, tournamentSize = 5):

        # Function to be optimized.
        self.function = function
        self.geneSize = nArgs
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        # Genetic Algorithm's parameters.
        self.maxIteractions = maxIteractions
        self.populationSize = populationSize
        self.threshold = threshold
        self.eliteNum = eliteNum
        self.crossoverNum = self.populationSize - self.eliteNum

        # Crossover parameters.
        self.crossover = crossoverMethod
        self.selectCouples = selectionMethod
        self.mutation = mutationMethod
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
        return np.random.choice(self.populationSize, size = (self.crossoverNum, 2), p = roulette)


    # Tournament selection.
    def tournamentSelect(self):
        tournament = np.random.randint(0, self.populationSize, (2*self.crossoverNum, self.tournamentSize))
        winners = np.min(tournament, axis = 1).reshape(self.crossoverNum, 2)
        return winners


    # Stochastic selection
    def stochasticSelect(self):
        fit = self.fitValues[-1] - self.fitValues
        distance = np.add.reduce(fit) / (2*self.crossoverNum)
        selected =  distance * (np.arange(2*self.crossoverNum) + random.random())
        parents = self.populationSize - np.add.reduce(selected[:, None] < np.cumsum(fit), axis = 1)
        return parents.reshape(self.crossoverNum, 2)


    """
    Methods of mutation (one of them must be chosen).
    """
    # This function has a chance of choosing each chromosome.
    # The chromosome chosen will have a random gene changed to a random value.
    def chromosomeMutation(self, population):
        elements = np.arange(self.crossoverNum)[np.random.rand(self.crossoverNum) < self.chromosomeMutationRate]
        positions = np.random.randint(0, self.geneSize, elements.size)
        population[elements, positions] = np.random.uniform(self.lowerBound[positions], self.upperBound[positions], elements.size)
        return population


    # This function has a chance of choosing each gene.
    # Every gene chosen will be changed to a random value.
    def geneMutation(self, population):
        mutation = np.random.rand(self.crossoverNum, self.geneSize) < self.geneMutationRate
        population[mutation] = np.random.uniform(self.lowerBound[None, :].repeat(self.crossoverNum, 0)[mutation],
                                                 self.upperBound[None, :].repeat(self.crossoverNum, 0)[mutation],
                                                 np.count_nonzero(mutation))
        return population


    """
    Methods linked to the crossover.
    """
    # Creates a boolean mask.
    def booleanMask(self, size):
        num = np.prod(size)
        numBytes = -(-num // 8)
        seqBytes = np.frombuffer(np.random.bytes(numBytes), np.uint8)
        return np.unpackbits(seqBytes)[:num].reshape(size)


    # Makes the crossover between two chromosomes randomly choosing the source of each gene.
    def uniformCrossover(self, couples):
        truth = self.booleanMask((couples.size // 2, self.geneSize))
        newPopulation = np.where(truth, self.population[couples[:, 0]], self.population[couples[:, 1]])
        return newPopulation


    # Makes the crossover between two chromosomes using genes from one parent before a random point and
    # from the other parent after that point.
    def singlePointCrossover(self, couples):
        grid = np.arange(self.geneSize)[None, :].repeat(couples.size // 2, 0)
        truth = grid < np.random.randint(0, self.geneSize + 1, couples.size // 2)[None, :].T
        newPopulation = np.where(truth, self.population[couples[:, 0]], self.population[couples[:, 1]])
        return newPopulation


    # Makes the crossover between two chromosomes using genes from one parent between two random points and
    # from the other parent outside the interval defined by the points.
    def twoPointCrossover(self, couples):
        grid = np.arange(self.geneSize)[None, :].repeat(couples.size // 2, 0)
        rand = np.sort(np.random.randint(0, self.geneSize + 1, (couples.size // 2, 2)))[None, :].T
        rand[0] -= 1
        truth = (grid >= rand[0]) & (grid < rand[1])
        newPopulation = np.where(truth, self.population[couples[:, 0]], self.population[couples[:, 1]])
        return newPopulation


    # Don't make any crossover.
    def noCrossover(self, couples):
        return self.population[couples[:, 0]]


    """
    Methods linked to the population and the generations.
    """
    def sortPopulation(self):
        argSort = self.fitValues.argsort()
        self.fitValues = self.fitValues[argSort]
        self.population = self.population[argSort]


    def firstPopulation(self):
        self.population = np.random.uniform(self.lowerBound, self.upperBound, (self.populationSize, self.geneSize))
        self.fitValues = self.function(self.population.T)
        self.sortPopulation()


    def nextGeneration(self):
        offspring = self.mutation(self, self.crossover(self, self.selectCouples(self)))
        self.population[self.eliteNum:] = offspring
        self.fitValues[self.eliteNum:] = self.function(offspring.T)
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

        # Returns the best chromosome on the population, since it's ordered.
        return self.population[0]


def GA(function, nArgs, lowerBound, upperBound, maxIteractions = 0, populationSize = 0, threshold = np.NINF,
        elitePercent = 0.05, tournamentSize = 0.1, chromosomeMutationRate = 0.2, geneMutationRate = 0.01,
        selectionMethod = None, mutationMethod = None, crossoverMethod = None):

    GA = geneticAlgorithm(function = function,
                          nArgs = nArgs,
                          lowerBound = lowerBound,
                          upperBound = upperBound,
                          maxIteractions = 100 * nArgs if maxIteractions == 0 else maxIteractions,
                          populationSize = min(20 * nArgs, 200) if populationSize == 0 else populationSize,
                          threshold = threshold,
                          eliteNum = math.ceil(populationSize * elitePercent),
                          chromosomeMutationRate = chromosomeMutationRate,
                          geneMutationRate = geneMutationRate,
                          tournamentSize = max(2, math.ceil(populationSize * tournamentSize)),
                          selectionMethod = selectionMethod if selectionMethod is not None else geneticAlgorithm.tournamentSelect,
                          mutationMethod = mutationMethod if mutationMethod is not None else geneticAlgorithm.geneMutation,
                          crossoverMethod = crossoverMethod if crossoverMethod is not None else geneticAlgorithm.crossoverUniform)
    return GA.run()
