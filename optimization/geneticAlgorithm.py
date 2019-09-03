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
        self.fArgs = kwargs.get('fArgs', {})
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
    Returns the indexes of the selected chromosomes.
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

        # Deterministic selection, based on the integer part of the relative values.
        intRule = intRule.cumsum()
        deterministicPoints = intRule.searchsorted(np.arange(intRule[-1]), 'right')

        # Stochastic selection, proportional to the fractional part of the relative values.
        fractionalRule /= np.add.reduce(fractionalRule)
        randomPoints = np.random.choice(self.populationSize, int(size - intRule[-1]), p = fractionalRule)

        return np.random.permutation(np.concatenate((deterministicPoints, randomPoints)))


    # No selection.
    # Just creates a list with random chromosomes in the population.
    def noSelect(self, size):
        return np.random.randint(0, self.populationSize, size)


    """
    Mutation selection (one of them must be chosen).
    Returns the indexes selected genes and their position.
    """
    # Choose random genes in the population.
    # Each gene has 'geneMutationRate' chance to be selected.
    def geneMutationBy(self, shape):
        mutationRate = self.parameters.get('geneMutationRate', 0.01)
        mask = np.random.rand(*shape) < mutationRate
        genePositions = mask.ravel().nonzero()[0] % shape[1]

        return mask, genePositions


    # Each chromosome in the population has 'chromosomeMutationRate' chance to be selected.
    # From each selected chromosome, one random gene will be chosen to be mutated.
    def chromosomeMutationBy(self, shape):
        mutationRate = self.parameters.get('chromosomeMutationRate', 0.75)
        chromosomes = (np.random.rand(shape[0]) < mutationRate).nonzero()[0]
        genePositions = np.random.randint(0, shape[1], chromosomes.size)

        return (chromosomes, genePositions), genePositions


    """
    Crossover methods (one of them must be chosen).
    """
    # Uniform Crossover
    # Generates two offspring from each pair of parents.
    def uniformCrossover(self):
        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        # Creates a random boolean mask with the same shape as the parents
        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-parent1.size // 8)), np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        # Second offspring will have the genes not selected on the first time
        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Single Point Crossover.
    # Generates two offspring from each pair of parents.
    def singlePointCrossover(self):
        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        rand = np.random.randint(0, parent1.shape[1] + 1, (parent1.shape[0], 1))
        mask = np.arange(parent1.shape[1]) < rand

        # Second offspring will have the genes not selected on the first time
        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Two Point Crossover
    # Generates two offspring from each pair of parents.
    def twoPointCrossover(self):
        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        grid = np.arange(parent1.shape[1])
        p1, p2 = np.random.randint(0, parent1.shape[1] + 1, (2, parent1.shape[0], 1))
        mask = (grid < p1) ^ (grid >= p2)

        # Second offspring will have the genes not selected on the first time
        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Discrete Crossover
    # Works like Uniform Crossover
    # Generates one offspring from each pair of parents.
    def discreteCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        # Creates a random boolean mask with the same shape as the parents
        mask = np.unpackbits(np.frombuffer(np.random.bytes(-(-parent1.size // 8)), np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        return np.where(mask, parent1, parent2)


    # Flat Crossover.
    # Each offspring's genes will be defined as a random number between the parents alleles.
    # This crossover will create new genes during the iterations.
    # Generates one offspring from each pair of parents.
    def flatCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]
        
        return parent1 + np.random.rand(*parent1.shape)*(parent2 - parent1)


    # Average Crossover.
    # Each offspring's genes will be defined as the average between the parents alleles.
    # This crossover will create new genes during the iterations.
    # Generates one offspring from each pair of parents.
    def averageCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        return (parent1 + parent2) / 2


    # No Crossover
    # Offsprings will be a clone of the selected parents.
    def noCrossover(self):
        return self.population[self.selection(self.crossoverSize)]


    """
    Mutation methods (one of them must be chosen).
    """
    # Uniform mutation.
    # Each selected gene will be changed to a new random value.
    def uniformMutation(self, population):
        mask, genePositions = self.mutationBy(population.shape)
        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        population[mask] = geneMin + np.random.rand(genePositions.size) * (geneMax - geneMin)
        return population


    # Creep Mutation
    # Each selected gene will be increased or decreased by a random value proportional to 'creepFactor.
    def creepMutation(self, population):
        mask, genePositions = self.mutationBy(population.shape)
        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        creepFactor = self.parameters.get('creepFactor', 0.001)
        geneRange = creepFactor*(geneMax - geneMin)

        population[mask] = (
            population[mask] + (2*np.random.rand(genePositions.size) - 1) * geneRange
        ).clip(geneMin, geneMax)

        return population


    # Gaussian Mutation
    # Eache gene will be changed to a new value given by a normal distribution
    # with mean equals the old gene value and standard deviation equals 'gaussianScale'.
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
    Population and the generations related methods.
    """
    # Sorts the population.
    def sortPopulation(self):
        argSort = self.values.argsort()
        self.values = self.values[argSort]
        self.population = self.population[argSort]


    # Generates the first population.
    def populate(self):
        self.population = self.lowerBound + np.random.rand(self.populationSize, self.geneSize) * (self.upperBound - self.lowerBound)
        self.values = self.fitness(self.population.T, **self.fArgs)
        self.sortPopulation()


    # Generates the next generation's population.
    def nextGeneration(self):
        offspring = self.mutation(self.crossover())
        self.population[self.eliteSize:] = offspring
        self.values[self.eliteSize:] = self.fitness(offspring.T, **self.fArgs)
        self.sortPopulation()

   
    """
    GA's execution.
    """
    # Runs the Generic Algorithm.
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
