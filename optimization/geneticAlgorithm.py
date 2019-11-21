import numpy as np
import math
import random


__all__ = ['GeneticAlgorithm', 'GA']


class GeneticAlgorithm:
    """
    Optimization class that implements Genetic Algorithm,
    which can be used to minimize positive functions.
    """

    def __init__(self, fitness, size, lowerBound, upperBound,
                 mutation = 'gaussian', selection = 'tournament', crossover = 'uniform',
                 dtype = np.float64, **kwargs):

        # Fitness function.
        self.fitness = fitness
        self.fArgs = kwargs.get('fArgs', {})
        self.geneSize = size
        self.dtype = dtype

        self.lowerBound = lowerBound.astype(dtype) if type(lowerBound) is np.ndarray \
                          else np.full(size, lowerBound, dtype)
        self.upperBound = upperBound.astype(dtype) if type(upperBound) is np.ndarray \
                          else np.full(size, upperBound, dtype)

        # Evolutionary methods.
        self.mutation = getattr(self, mutation + 'Mutation')
        self.selection = getattr(self, selection + 'Select')
        self.crossover = getattr(self, crossover + 'Crossover')
        self.mutationBy = getattr(self, kwargs.get('mutationBy', 'gene') + 'MutationBy')

        # Evolutionary parameters.
        self.maxGenerations = kwargs.get('maxGenerations', 100 * self.geneSize)
        self.populationSize = kwargs.get('populationSize', 200)
        self.eliteSize = kwargs.get('eliteSize', self.populationSize // 20)
        self.threshold = kwargs.get('threshold', np.NINF)
        self.crossoverSize = self.populationSize - self.eliteSize

        # Other parameters.
        self.parameters = kwargs


    """
    Selection methods (one of them must be chosen).
    Returns the indexes of the selected chromosomes.
    """

    # Tournament Selection. (Recommended)
    def tournamentSelect(self, size):
        try:
            tournamentShape = (self.parameters['tournamentSize'], size)
        except KeyError:
            self.parameters['tournamentSize'] = self.populationSize // 20
            tournamentShape = (self.parameters['tournamentSize'], size)

        tournament = np.random.randint(0, self.populationSize, tournamentShape)
        return tournament.min(axis = 0)


    # Rank Selection.
    def rankSelect(self, size):
        rank = np.arange(self.populationSize, 0, -1)
        rank /= self.populationSize * (self.populationSize + 1) / 2
        return np.random.choice(self.populationSize, size, p = rank)


    # Roulette Wheel Selection.
    def wheelSelect(self, size):
        roulette = self.values[-1] - self.values
        roulette /= np.add.reduce(roulette)
        return np.random.choice(self.populationSize, size, p = roulette)


    # Stochastic Universal Selection (SUS).
    def stochasticUniversalSelect(self, size):
        rule = (self.values[-1] - self.values).cumsum()
        distance = rule[-1] / size
        points = rule.searchsorted(distance * np.arange(random.random(), size))
        np.random.shuffle(points)
        return points


    # No selection.
    def noSelect(self, size):
        """
        Just returns a vector of random chromosomes in the population.
        """
        return np.random.randint(0, self.populationSize, size)


    """
    Mutation selection (one of them must be chosen).
    Returns the indexes selected genes and their position.
    """

    # Gene Mutation.
    def geneMutationBy(self, shape):
        """
        Choose random genes in the population.
        Each gene has 'geneMutationRate' chance to be selected.
        """

        try:
            mutationRate = self.parameters['geneMutationRate']
        except KeyError:
            mutationRate = self.parameters['geneMutationRate'] = 0.01

        mask = np.random.rand(*shape) < mutationRate
        genePositions = mask.ravel().nonzero()[0] % shape[1]

        return mask, genePositions


    # Chromosome Mutation.
    def chromosomeMutationBy(self, shape):
        """
        Each chromosome in the population has 'chromosomeMutationRate' chance
        to be selected. From each selected chromosome, one random gene will be
        chosen to be mutated.
        """

        try:
            mutationRate = self.parameters['chromosomeMutationRate']
        except KeyError:
            mutationRate = self.parameters['chromosomeMutationRate'] = 0.75

        chromosomes = (np.random.rand(shape[0]) < mutationRate).nonzero()[0]
        genePositions = np.random.randint(0, shape[1], chromosomes.size)

        return (chromosomes, genePositions), genePositions


    """
    Crossover methods (one of them must be chosen).
    """
    
    # Uniform Crossover
    def uniformCrossover(self):
        """
        Generates two offspring from each pair of parents.
        """

        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        # Creates a random boolean mask with the same shape as the parents
        mask = np.unpackbits(np.frombuffer(np.random.bytes(math.ceil(parent1.size/8)), np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        # Second offspring will have the genes not selected on the first time
        return np.concatenate((np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)))


    # Discrete Crossover
    # Works like Uniform Crossover
    # Generates one offspring from each pair of parents.
    def discreteCrossover(self):
        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        # Creates a random boolean mask with the same shape as the parents
        mask = np.unpackbits(np.frombuffer(np.random.bytes(math.ceil(parent1.size/8)), np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        return np.where(mask, parent1, parent2)


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
        self.population = self.lowerBound + (
            np.random.rand(self.populationSize, self.geneSize) * (self.upperBound - self.lowerBound)
        ).astype(self.dtype)
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


# Runs the Genetic Algorithms.
def GA(fitness, size, lowerBound, upperBound, dtype = np.float64,
       mutation = 'gaussian', selection = 'tournament', crossover = 'uniform',
       **kwargs):
    """
    Simple caller for GeneticAlgorithm.
    """

    GA = GeneticAlgorithm(
        fitness = fitness,
        size = size,
        lowerBound = lowerBound,
        upperBound = upperBound,
        dtype = dtype,

        selection = selection,
        mutation = mutation,
        crossover = crossover,
        **kwargs
    )

    return GA.run()
