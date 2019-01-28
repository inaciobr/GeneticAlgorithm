"""
Genetic Algorithm used to minimize a positive function.
"""

import numpy as np

class geneticAlgorithm:
    def __init__(self, function, nArgs, lowerBound, upperBound, maxIteractions, populationSize, threshold, eliteNum,
                 selectionMethod, mutationMethod, chromosomeMutationRate = 0.2, geneMutationRate = 0.01, tournamentSize = 5):

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
        tournament = np.random.randint(0, self.populationSize, (self.crossoverNum << 1, self.tournamentSize))
        winners = tournament[np.arange(self.crossoverNum << 1), tournament.argmin(axis = 1)]
        return winners.reshape(self.crossoverNum, 2)


    """
    Methods of mutation (one of them must be chosen).
    """
    # This function has a chance of choosing each chromosome.
    # The chromosome chosen will have a random gene changed to a random value.
    def chromosomeMutation(self, population):
        elements = np.arange(self.crossoverNum)[np.random.rand(self.crossoverNum) <= self.chromosomeMutationRate]
        positions = np.random.randint(0, self.geneSize, elements.size)
        population[elements, positions] = np.random.uniform(self.lowerBound[positions], self.upperBound[positions], elements.size)
        return population


    # This function has a chance of choosing each gene.
    # Every gene chosen will be changed to a random value.
    def geneMutation(self, population):
        mutation = np.random.rand(self.crossoverNum, self.geneSize) <= self.geneMutationRate
        population[mutation] = np.random.uniform(self.lowerBound.reshape(1, -1).repeat(self.crossoverNum, 0)[mutation],
                                                 self.upperBound.reshape(1, -1).repeat(self.crossoverNum, 0)[mutation],
                                                 np.count_nonzero(mutation))
        return population


    """
    Methods linked to the crossover.
    """
    # Creates a boolean mask.
    def booleanMask(self, size):
        num = np.prod(size)
        numBytes = -(-num >> 3)
        seqBytes = np.frombuffer(np.random.bytes(numBytes), np.uint8)
        return np.unpackbits(seqBytes)[:num].reshape(size)


    # Makes the crossover between couples of the same population.
    def crossover(self, couples):
        truth = self.booleanMask((couples.size >> 1, self.geneSize))
        newPopulation = np.where(truth, self.population[couples[:, 0]], self.population[couples[:, 1]])
        return self.mutation(self, newPopulation)


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
        offspring = self.crossover(self.selectCouples(self))
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
