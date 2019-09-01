import numpy as np
import math
import random


__all__ = ['GeneticAlgorithm', 'GA']


"""
Genetic Algorithm used to minimize positive functions.
"""
class GeneticAlgorithm:
    def __init__(self, fitness, size, lowerBound, upperBound,
                 mutation = 'gaussian', selection = 'tournament', crossover = 'uniform',
                 **kwargs):

        # Function to be optimized.
        self.fitness = fitness
        self.funcParams = kwargs.get('funcParams', {})
        self.geneSize = size
        self.lowerBound = lowerBound if type(lowerBound) is np.ndarray else np.array([lowerBound] * size)
        self.upperBound = upperBound if type(upperBound) is np.ndarray else np.array([upperBound] * size)

        # Genetic Algorithm methods.
        self.mutation = getattr(self, mutation + 'Mutation')
        self.selection = getattr(self, selection + 'Select')
        self.crossover = getattr(self, crossover + 'Crossover')
        self.mutationBy = getattr(self, kwargs.get('mutationBy', 'gene') + 'MutationBy')

        # Generations Parameters.
        self.maxGenerations = kwargs.get('maxGenerations', 100 * self.geneSize)
        self.threshold = kwargs.get('threshold', np.NINF)

        self.populationSize = kwargs.get('populationSize', 200)
        self.eliteSize = kwargs.get('eliteSize', self.populationSize // 20)
        self.crossoverSize = self.populationSize - self.eliteSize

        # Method parameters.
        self.parameters = kwargs


    """
    Selection methods (one of them must be chosen).
    """
    # Tournament selection.
    # Recommended.
    def tournamentSelect(self, size):
        tournamentSize = self.parameters.get('tournamentSize', self.populationSize // 20)
        tournament = np.random.randint(0, self.populationSize, (tournamentSize, size))
        return tournament.min(axis = 0)


    # Roulette Wheel selection.
    def wheelSelect(self, size):
        roulette = self.values[-1] - self.values
        roulette /= np.add.reduce(roulette)
        return np.random.choice(self.populationSize, size, p = roulette)


    # Rank selection.
    def rankSelect(self, size):
        rank = np.arange(self.populationSize, 0, -1)
        rank /= self.populationSize * (self.populationSize + 1) / 2
        return np.random.choice(self.populationSize, size, p = rank)


    # Stochastic Universal selection.
    def stochasticUniversalSelect(self, size):
        rule = (self.values[-1] - self.values).cumsum()
        distance = rule[-1] / size
        points = rule.searchsorted(distance * np.arange(random.random(), size))
        return np.random.permutation(points)


    # Stochastic Remainder selection.
    def stochasticRemainderSelect(self, size):
        rule = (self.values[-1] - self.values)
        fractionalRule, intRule = np.modf(rule / rule.mean())

        intRule = intRule.cumsum()
        deterministicPoints = intRule.searchsorted(np.arange(intRule[-1]), 'right')

        fractionalRule /= np.add.reduce(fractionalRule)
        randomPoints = np.random.choice(self.populationSize, int(size - intRule[-1]), p = fractionalRule)

        return np.random.permutation(np.concatenate((deterministicPoints, randomPoints)))


    # Uniform selection.
    def noSelect(self, size):
        return np.random.randint(0, self.populationSize, size)


    """
    Crossover methods (one of them must be chosen).
    """
    # Does the crossover between two chromosomes randomly choosing the source of each gene.
    # Recommended.
    def uniformCrossover(self):
        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-parent1.size // 8)), np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Does the crossover between two chromosomes using genes from one parent
    # until a random point and from the other parent after that point.
    def singlePointCrossover(self):
        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        rand = np.random.randint(0, parent1.shape[1] + 1, (parent1.shape[0], 1))
        mask = np.arange(parent1.shape[1]) < rand
        
        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Does the crossover between two chromosomes using genes from one parent between two random points
    # and from the other parent outside the interval defined by the points.
    def twoPointCrossover(self):
        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        grid = np.arange(parent1.shape[1])
        p1, p2 = np.random.randint(0, parent1.shape[1] + 1, (2, parent1.shape[0], 1))
        mask = (grid < p1) ^ (grid >= p2)

        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Does the crossover the same way uniformCrossover does, but generates only one child
    # per couple.
    def discreteCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-parent1.size // 8)), np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        return np.where(mask, parent1, parent2)


    # Does the crossover between two chromosomes using a random value between the minimum and maximum
    # values of each allele.
    def flatCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]
        
        return parent1 + np.random.rand(*parent1.shape)*(parent2 - parent1)


    # Does the crossover between two chromosomes using the average value between alleles.
    def averageCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        return (parent1 + parent2) / 2


    # Don't do any crossover.
    def noCrossover(self):
        return self.population[self.selection(self.crossoverSize)]


    """
    Mutation selection (one of them must be chosen).
    """
    # Choose random genes in the population.
    def geneMutationBy(self, shape):
        mutationRate = self.parameters.get('geneMutationRate', 0.01)
        mask = np.random.rand(*shape) < mutationRate
        genePositions = mask.ravel().nonzero()[0] % shape[1]

        return mask, genePositions


    # Choose random chromosomes in the population and from
    # each of the chosen ones, a random gene is selected.
    def chromosomeMutationBy(self, shape):
        mutationRate = self.parameters.get('chromosomeMutationRate', 0.75)
        chromosomes = (np.random.rand(shape[0]) < mutationRate).nonzero()[0]
        genePositions = np.random.randint(0, shape[1], chromosomes.size)

        return (chromosomes, genePositions), genePositions


    """
    Mutation methods (one of them must be chosen).
    """
    # Every gene chosen will be changed to a random value.
    def uniformMutation(self, population):
        mask, genePositions = self.mutationBy(population.shape)
        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        population[mask] = np.random.uniform(geneMin, geneMax, genePositions.size)
        return population


    # Every gene chosen will be increased or decreased by a random value
    # between 0 and the range between maximum and minimum possible
    def creepMutation(self, population):
        mask, genePositions = self.mutationBy(population.shape)
        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        creepFactor = self.parameters.get('creepFactor', 0.001)
        geneRange = creepFactor*(geneMax - geneMin)

        population[mask] = (
            population[mask] + np.random.uniform(-geneRange, geneRange, genePositions.size)
        ).clip(geneMin, geneMax)

        return population


    # Every gene chosen will have a value from a gaussian distribution added to it.
    def gaussianMutation(self, population):
        mask, genePositions = self.mutationBy(population.shape)
        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        gaussianScale = self.parameters.get('gaussianScale', 1.0)

        population[mask] = (
            population[mask] + np.random.normal(scale = gaussianScale, size = genePositions.size)
        ).clip(geneMin, geneMax)

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
        self.values = self.fitness(self.population.T, **self.funcParams)
        self.sortPopulation()


    # Generates the next generation's population.
    def nextGeneration(self):
        offspring = self.mutation(self.crossover())
        self.population[self.eliteSize:] = offspring
        self.values[self.eliteSize:] = self.fitness(offspring.T, **self.funcParams)
        self.sortPopulation()

   
    """
    GA's execution.
    """
    def run(self):
        self.populate()

        for _ in range(self.maxGenerations):
            if self.values[0] <= self.threshold:
                break

            self.nextGeneration()

        # Returns the chromosome with best fit on the population, since it's ordered.
        return self.population[0], self.values[0]


    # Plot GA's evolution over the generations.
    def graph(self):
        import matplotlib.pyplot as plt
        values = np.zeros(self.maxGenerations)
        self.populate()

        for i in range(self.maxGenerations):
            self.nextGeneration()
            values[i] = self.values[0]

        plt.plot(np.arange(self.maxGenerations), values)
        plt.show()


"""
Simple caller for GeneticAlgorithm.
"""
# Runs the Genetic Algorithms.
def GA(fitness, size, lowerBound, upperBound,
    mutation = 'gaussian', selection = 'tournament', crossover = 'uniform',
    **kwargs):

    GA = GeneticAlgorithm(
        fitness = fitness,
        size = size,
        lowerBound = lowerBound,
        upperBound = upperBound,

        selection = selection,
        mutation = mutation,
        crossover = crossover,
        **kwargs
    )

    return GA.run()
