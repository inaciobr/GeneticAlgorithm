import numpy as np
import math
import random

"""
Genetic Algorithm used to minimize positive functions.
"""
class GeneticAlgorithm:
    def __init__(self, fitnessFunction, geneSize, lowerBound, upperBound,
                 maxIteractions, populationSize, eliteSize,
                 threshold = np.NINF, selectionMethod = None, mutationMethod = None, crossoverMethod = None,
                 chromosomeMutationRate = 0.2, geneMutationRate = 0.01, tournamentSize = 5):

        # Function to be optimized.
        self.geneSize = geneSize
        self.fitnessFunction = fitnessFunction
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        # Parameters of GeneticAlgorithm.
        self.maxGenerations = maxIteractions
        self.populationSize = populationSize
        self.eliteSize = eliteSize
        self.threshold = threshold

        # Methods of GeneticAlgorithm.
        self.selectCouples = selectionMethod if selectionMethod else GeneticAlgorithm.tournamentSelect
        self.mutation = mutationMethod if mutationMethod else GeneticAlgorithm.geneMutation
        self.crossover = crossoverMethod if crossoverMethod else GeneticAlgorithm.singlePointCrossover

        # Parameters of methods.
        self.crossoverNum = self.populationSize - self.eliteSize
        self.chromosomeMutationRate = chromosomeMutationRate
        self.geneMutationRate = geneMutationRate
        self.tournamentSize = tournamentSize


    """
    Methods of selection of couples (one of them must be chosen).
    """
    # Tournament selection.
    def tournamentSelect(self):
        tournament = np.random.randint(0, self.populationSize, (self.tournamentSize, 2, self.crossoverNum))
        return tournament.min(axis = 0)


    # Roulette Wheel selection.
    def wheelSelect(self):
        roulette = self.fitValues[-1] - self.fitValues
        return np.random.choice(self.populationSize, size = (2, self.crossoverNum), p = roulette / np.add.reduce(roulette))


    # *Stochastic selection. (Can be better)
    def stochasticSelect(self):
        cumsum = np.cumsum(self.fitValues[-1] - self.fitValues)
        distance = cumsum[-1] / self.crossoverNum
        points = np.random.permutation(distance * np.arange(random.random(), 2*self.crossoverNum) / 2.)
        return np.searchsorted(cumsum, points).reshape(2, self.crossoverNum)


    """
    Methods of crossover.
    """
    # Makes the crossover between two chromosomes randomly choosing the source of each gene.
    def uniformCrossover(self, couples):
        height = couples.size // 2
        size = height * self.geneSize
        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-size//8)), np.uint8))[:size].reshape(height, self.geneSize)
        return np.where(mask, *self.population[couples])


    # Makes the crossover between two chromosomes using genes from one parent before a random point and
    # from the other parent after that point.
    def singlePointCrossover(self, couples):
        mask = np.arange(self.geneSize) < np.random.randint(0, self.geneSize + 1, couples.size // 2)[:, None]
        return np.where(mask, *self.population[couples])


    # *Makes the crossover between two chromosomes using genes from one parent between two random points and
    # from the other parent outside the interval defined by the points. (Can be better)
    def twoPointCrossover(self, couples):
        grid = np.arange(self.geneSize)
        rand = np.sort(np.random.randint(0, self.geneSize + 1, (2, couples.size // 2, 1)))
        rand[0] -= 1
        mask = (grid >= rand[0]) & (grid < rand[1])
        return np.where(mask, *self.population[couples])


    # Don't make any crossover.
    def noCrossover(self, couples):
        return self.population[couples[0]] 


    """
    Methods of mutation (one of them must be chosen).
    """
    # This function has a chance of choosing each chromosome.
    # The chromosome chosen will have a random gene changed to a random value.
    def chromosomeMutation(self, population):
        elements = np.nonzero(np.random.rand(population.shape[0]) < self.chromosomeMutationRate)[0]
        positions = np.random.randint(0, population.shape[1], elements.size)
        population[elements, positions] = np.random.uniform(self.lowerBound[positions], self.upperBound[positions], positions.size)
        return population


    # This function has a chance of choosing each gene.
    # Every gene chosen will be changed to a random value.
    def geneMutation(self, population):
        mask = np.random.rand(*population.shape) < self.geneMutationRate
        positions = mask.ravel().nonzero()[0] % population.shape[1]
        population[mask] = np.random.uniform(self.lowerBound[positions], self.upperBound[positions], positions.size)
        return population


    """
    Methods related to the population and the generations.
    """
    def sortPopulation(self):
        argSort = self.fitValues.argsort()
        self.fitValues = self.fitValues[argSort]
        self.population = self.population[argSort]


    def firstPopulation(self):
        self.population = np.random.uniform(self.lowerBound, self.upperBound, (self.populationSize, self.geneSize))
        self.fitValues = self.fitnessFunction(self.population.T)
        self.sortPopulation()


    def nextGeneration(self):
        offspring = self.mutation(self, self.crossover(self, self.selectCouples(self)))
        self.population[self.eliteSize:] = offspring
        self.fitValues[self.eliteSize:] = self.fitnessFunction(offspring.T)
        self.sortPopulation()

   
    """
    Iterates over the generations.
    """
    def run(self):
        self.firstPopulation()

        for _ in range(self.maxGenerations):
            if self.fitValues[0] < self.threshold:
                break

            self.nextGeneration()

        # Returns the chromosome with best fit on the population, since it's ordered.
        return self.population[0]
