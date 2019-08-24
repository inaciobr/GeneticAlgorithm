import numpy as np
import math
import random


__all__ = ['GeneticAlgorithm']


"""
Genetic Algorithm used to minimize positive functions.
"""
class GeneticAlgorithm:
    def __init__(self, fitness, size, lowerBound, upperBound,
                 maxGenerations = None, threshold = np.NINF,
                 populationSize = 200, eliteSize = 10, mutationRate = 0.01,
                 selection = 'tournament', mutation = 'gene', crossover = 'uniform',
                 **kwargs):

        # Function to be optimized.
        self.fitness = fitness
        self.geneSize = size
        self.lowerBound = lowerBound if type(lowerBound) is np.ndarray else np.array([lowerBound] * size)
        self.upperBound = upperBound if type(upperBound) is np.ndarray else np.array([upperBound] * size)

        # Generations Parameters.
        self.maxGenerations = maxGenerations if maxGenerations else 100*size
        self.threshold = threshold

        # Genetic Algorithm methods.
        self.selection = getattr(self, selection + 'Select')
        self.mutation = getattr(self, mutation + 'Mutation')
        self.crossover = getattr(self, crossover + 'Crossover')

        # Population Parameters.
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.eliteSize = eliteSize
        self.crossoverSize = self.populationSize - self.eliteSize

        # Method parameters.
        self.parameters = kwargs


    """
    Selection methods (one of them must be chosen).
    """
    # Tournament selection.
    # Recommended.
    def tournamentSelect(self, size):
        tournamentSize = self.parameters.get('tournamentSize', 10)
        tournament = np.random.randint(0, self.populationSize, (tournamentSize, size))
        return tournament.min(axis = 0)


    # Roulette Wheel selection.
    def wheelSelect(self, size):
        roulette = self.values[-1] - self.values
        return np.random.choice(self.populationSize, size, p = roulette / np.add.reduce(roulette))


    # Rank selection.
    def rankSelect(self, size):
        rank = np.arange(self.populationSize, 0, -1)
        total = self.populationSize * (self.populationSize + 1) / 2
        return np.random.choice(self.populationSize, size, p = rank / total)


    # Stochastic selection.
    def stochasticSelect(self, size):
        rule = (self.values[-1] - self.values).cumsum()
        distance = rule[-1] / size
        points = rule.searchsorted(distance * np.arange(random.random(), size))
        np.random.shuffle(points)        
        return points


    """
    Crossover methods (one of them must be chosen).
    """
    # Does the crossover between two chromosomes randomly choosing the source of each gene.
    # Recommended.
    def uniformCrossover(self):
        parents = self.population[self.selection(self.crossoverSize).reshape(2, -1)]
        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-parents[0].size // 8)), np.uint8))
        mask = mask[:parents[0].size].reshape(*parents[0].shape)
        return np.concatenate((np.where(mask, *parents), np.where(~mask, *parents)))


    # Does the crossover between two chromosomes using genes from one parent
    # until a random point and from the other parent after that point.
    def singlePointCrossover(self):
        parents = self.population[self.selection(self.crossoverSize).reshape(2, -1)]
        mask = np.arange(parents[0].shape[1]) < np.random.randint(0, parents[0].shape[1] + 1, (parents.shape[1], 1))
        return np.concatenate((np.where(mask, *parents), np.where(~mask, *parents)))


    # Does the crossover between two chromosomes using genes from one parent between two random points
    # and from the other parent outside the interval defined by the points.
    def twoPointCrossover(self):
        parents = self.population[self.selection(self.crossoverSize).reshape(2, -1)]
        grid = np.arange(parents[0].shape[1])
        rand = np.random.randint(0, parents[0].shape[1] + 1, (2, parents.shape[1], 1))
        mask = (grid < rand[0]) ^ (grid > rand[1] - 1)
        return np.concatenate((np.where(mask, *parents), np.where(~mask, *parents)))


    # Does the crossover the same way uniformCrossover does, but generates only one child
    # per couple.
    def discreteCrossover(self):
        parents = self.population[self.selection(2*self.crossoverSize).reshape(2, self.crossoverSize)]
        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-parents[0].size // 8)), np.uint8))
        mask = mask[:parents[0].size].reshape(*parents[0].shape)
        return np.where(mask, *parents)


    # Does the crossover between two chromosomes using the average value between alleles.
    def averageCrossover(self):
        parents = self.population[self.selection(2*self.crossoverSize).reshape(2, self.crossoverSize)]
        return (parents[0] + parents[1]) / 2


    # Does the crossover between two chromosomes using a random value between the minimum and maximum
    # values of each allele.
    def flatCrossover(self):
        parents = self.population[self.selection(2*self.crossoverSize).reshape(2, self.crossoverSize)]
        r = np.random.rand(self.crossoverSize, self.geneSize)
        return r*parents[0] + (1 - r)*parents[1]


    # Don't do any crossover.
    def noCrossover(self):
        return self.population[self.selection(self.crossoverSize)]


    """
    Mutation methods (one of them must be chosen).
    """
    # This function has a chance of choosing each chromosome.
    # The chromosome chosen will have a random gene changed to a random value.
    def chromosomeMutation(self, population):
        elements = (np.random.rand(population.shape[0]) < self.mutationRate).nonzero()[0]
        positions = np.random.randint(0, population.shape[1], elements.size)
        population[elements, positions] = np.random.uniform(self.lowerBound[positions], self.upperBound[positions], positions.size)
        return population


    # This function has a chance of choosing each gene.
    # Every gene chosen will be changed to a random value.
    # Uniform Crossover
    def geneMutation(self, population):
        mask = np.random.rand(*population.shape) < self.mutationRate
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
        offspring = self.mutation(self.crossover())
        self.population[self.eliteSize:] = offspring
        self.values[self.eliteSize:] = self.fitness(offspring.T)
        self.sortPopulation()

   
    """
    GA's execution.
    """
    def run(self):
        self.populate()

        iters = self.maxGenerations
        while iters and self.threshold < self.values[0]:
            self.nextGeneration()
            iters -= 1

        # Returns the chromosome with best fit on the population, since it's ordered.
        return self.population[0]
