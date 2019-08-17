import numpy as np
import math
import random

"""
Genetic Algorithm used to minimize positive functions.
"""
class GeneticAlgorithm:
    def __init__(self, fitness, size, lowerBound, upperBound, populationSize,
                 maxIteractions, eliteSize, threshold = np.NINF, 
                 selectionMethod = None, mutationMethod = None, crossoverMethod = None,
                 chromosomeMutationRate = 0.2, geneMutationRate = 0.01, tournamentSize = 5):

        # Function to be optimized.
        self.fitness = fitness
        self.geneSize = size
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        # Parameters of GeneticAlgorithm.
        self.populationSize = populationSize
        self.maxGenerations = maxIteractions
        self.eliteSize = eliteSize
        self.threshold = threshold

        # Methods of GeneticAlgorithm.
        self.selection = selectionMethod if selectionMethod else GeneticAlgorithm.tournamentSelect
        self.mutation = mutationMethod if mutationMethod else GeneticAlgorithm.geneMutation
        self.crossover = crossoverMethod if crossoverMethod else GeneticAlgorithm.singlePointCrossover

        # Parameters of methods.
        self.crossoverSize = self.populationSize - self.eliteSize
        self.chromosomeMutationRate = chromosomeMutationRate
        self.geneMutationRate = geneMutationRate
        self.tournamentSize = tournamentSize


    """
    Methods of selection of parents (one of them must be chosen).
    """
    # Tournament selection.
    def tournamentSelect(self):
        tournament = np.random.randint(0, self.populationSize, (self.tournamentSize, 2, self.crossoverSize))
        return tournament.min(axis = 0)


    # Roulette Wheel selection.
    def wheelSelect(self):
        roulette = self.values[-1] - self.values
        return np.random.choice(self.populationSize, (2, self.crossoverSize), p = roulette / np.add.reduce(roulette))


    # Rank selection.
    def rankSelect(self):
        rank = np.arange(self.populationSize, 0, -1)
        total = self.populationSize * (self.populationSize + 1) / 2
        return np.random.choice(self.populationSize, (2, self.crossoverSize), p = rank / total)


    # Stochastic selection.
    def stochasticSelect(self):
        rule = (self.values[-1] - self.values).cumsum()
        distance = rule[-1] / (2*self.crossoverSize)
        points = rule.searchsorted(distance * np.arange(random.random(), 2*self.crossoverSize))
        np.random.shuffle(points)        
        return points.reshape(2, self.crossoverSize)


    """
    Methods of crossover.
    """
    # Don't make any crossover.
    def noCrossover(self, parents):
        return self.population[parents[0]]


    # Makes the crossover between two chromosomes using genes from one parent before a random point and
    # from the other parent after that point.
    def singlePointCrossover(self, parents):
        mask = np.arange(self.geneSize) < np.random.randint(0, self.geneSize + 1, (parents.size // 2, 1))
        return np.where(mask, *self.population[parents])


    # *Makes the crossover between two chromosomes using genes from one parent between two random points and
    # from the other parent outside the interval defined by the points. (Can be better)
    def twoPointCrossover(self, parents):
        grid = np.arange(self.geneSize)
        rand = np.sort(np.random.randint(0, self.geneSize + 1, (2, parents.size // 2, 1)))
        rand[0] -= 1
        mask = (grid >= rand[0]) & (grid < rand[1])
        return np.where(mask, *self.population[parents])


    # Makes the crossover between two chromosomes randomly choosing the source of each gene.
    def uniformCrossover(self, parents):
        height = parents.size // 2
        size = height * self.geneSize
        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-size//8)), np.uint8))[:size].reshape(height, self.geneSize)
        return np.where(mask, *self.population[parents])


    """
    Methods of mutation (one of them must be chosen).
    """
    # This function has a chance of choosing each chromosome.
    # The chromosome chosen will have a random gene changed to a random value.
    def chromosomeMutation(self, population):
        elements = (np.random.rand(population.shape[0]) < self.chromosomeMutationRate).nonzero()[0]
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
    # Sorts the population.
    def sortPopulation(self):
        argSort = self.values.argsort()
        self.values = self.values[argSort]
        self.population = self.population[argSort]


    # Generates the first population.
    def populate(self):
        self.population = np.random.uniform(self.lowerBound, self.upperBound, (self.populationSize, self.geneSize))
        self.values = self.fitness(self.population.T)
        self.sortPopulation()


    # Generates the next generation's population.
    def nextGeneration(self):
        offspring = self.mutation(self, self.crossover(self, self.selection(self)))
        self.population[self.eliteSize:] = offspring
        self.values[self.eliteSize:] = self.fitness(offspring.T)
        self.sortPopulation()

   
    """
    Iterates over the generations.
    """
    def run(self):
        self.populate()

        iters = self.maxGenerations
        while iters and self.threshold < self.values[0]:
            self.nextGeneration()
            iters -= 1

        # Returns the chromosome with best fit on the population, since it's ordered.
        return self.population[0]
