import numpy as np
import math
import random


__all__ = ['GeneticAlgorithm', 'GA']


class GeneticAlgorithm:
    """
    Implementation of the Genetic Algorithm.
    Can be used to minimize positive functions.
    """

    def __init__(self, fitness, size, lowerBound, upperBound,
                 mutation='gaussian', selection='tournament',
                 crossover='uniform', dtype=np.float64, **kwargs):

        # Fitness function.
        self.fitness = fitness
        self.fArgs = kwargs.get('fArgs', {})
        self.geneSize = size
        self.dtype = dtype

        self.lowerBound = (
            lowerBound.astype(dtype) if type(lowerBound) is np.ndarray
            else np.full(size, lowerBound, dtype)
        )

        self.upperBound = (
            upperBound.astype(dtype) if type(upperBound) is np.ndarray
            else np.full(size, upperBound, dtype)
        )

        # Evolutionary methods.
        self.mutation = getattr(self, mutation + 'Mutation')
        self.selection = getattr(self, selection + 'Select')
        self.crossover = getattr(self, crossover + 'Crossover')
        self.mutationBy = getattr(self, kwargs.get('mutationBy', 'gene') 
                                  + 'MutationBy')

        # Evolutionary parameters.
        self.maxGenerations = kwargs.get('maxGenerations', 100 * self.geneSize)
        self.populationSize = kwargs.get('populationSize', 200)
        self.eliteSize = kwargs.get('eliteSize', self.populationSize // 20)
        self.threshold = kwargs.get('threshold', np.NINF)
        self.crossoverSize = self.populationSize - self.eliteSize

        # Other parameters.
        self.getNormalizedValues = self.distanceToLeast
        self.parameters = kwargs


    # Normalization methods to be used when necessary.
    def distanceToLeast(self):
        return self.values[self.argSort[-1]] - self.values


    # Selection methods (one of them must be chosen).
    # Returns the indexes of the selected chromosomes.

    # Recommended
    def tournamentSelect(self, size):
        try:
            tournamentSize = self.parameters['tournamentSize']
        except KeyError:
            tournamentSize = self.parameters['tournamentSize'] \
                           = self.populationSize // 20

        winners = np.random.randint(0, self.populationSize,
                                    (tournamentSize, size)).min(axis=0)

        return self.argSort[winners]


    def rankSelect(self, size):
        try:
            rank = self.rank
        except AttributeError:
            norm = self.populationSize * (self.populationSize + 1) / 2
            rank = self.rank = np.arange(self.populationSize, 0, -1.0) / norm

        points = np.random.choice(self.populationSize, size, p = rank)
        return self.argSort[points]


    def wheelSelect(self, size):
        roulette = self.getNormalizedValues()
        roulette /= np.add.reduce(roulette)

        points = np.random.choice(self.populationSize, size, p = roulette)
        return self.argSort[points]


    def stochasticUniversalSelect(self, size):
        rule = self.getNormalizedValues().cumsum()
        distance = rule[-1] / size

        points = rule.searchsorted(distance * np.arange(random.random(), size))
        np.random.shuffle(points)

        return self.argSort[points]


    def noSelect(self, size):
        """
        Just returns a vector of random chromosomes in the population.
        """
        
        return np.random.randint(0, self.populationSize, size)


    """
    Mutation selection (one of them must be chosen).
    Returns the indexes selected genes and their position.
    """

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

    def uniformCrossover(self):
        """
        Generates two offspring from each pair of parents.
        """

        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        # Creates a random boolean mask with the same shape as the parents
        randomBytes = np.random.bytes(math.ceil(parent1.size / 8))
        mask = np.unpackbits(np.frombuffer(randomBytes, np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        # Second offspring will have the genes not selected on the first time
        gen1 = np.where(mask, parent1, parent2)
        gen2 = np.where(mask, parent2, parent1)

        return np.concatenate((gen1, gen2))


    def discreteCrossover(self):
        """
        Works like Uniform Crossover, but generates only one offspring from each
        pair of parents.
        """

        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        # Creates a random boolean mask with the same shape as the parents
        randomBytes = np.random.bytes(math.ceil(parent1.size / 8))
        mask = np.unpackbits(np.frombuffer(randomBytes, np.uint8))
        mask = mask[:parent1.size].reshape(*parent1.shape)

        return np.where(mask, parent1, parent2)


    def singlePointCrossover(self):
        """
        Generates two offspring from each pair of parents.
        """

        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        rand = np.random.randint(0, parent1.shape[1] + 1, (parent1.shape[0], 1))
        mask = np.arange(parent1.shape[1]) < rand

        # Second offspring will have the genes not selected on the first time
        gen1 = np.where(mask, parent1, parent2)
        gen2 = np.where(mask, parent2, parent1)

        return np.concatenate((gen1, gen2))


    def twoPointCrossover(self):
        """
        Generates two offspring from each pair of parents.
        """

        select = self.selection(self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        grid = np.arange(parent1.shape[1])
        p1, p2 = np.random.randint(0, parent1.shape[1] + 1, 
                                   (2, parent1.shape[0], 1))
        mask = (grid < p1) ^ (grid >= p2)

        # Second offspring will have the genes not selected on the first time
        gen1 = np.where(mask, parent1, parent2)
        gen2 = np.where(mask, parent2, parent1)
        
        return np.concatenate((gen1, gen2))


    def flatCrossover(self):
        """
        Each offspring's genes will be defined as a random number between the 
        parents alleles.
        This crossover will create new genes during the iterations.
        Generates one offspring from each pair of parents.
        """

        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]
        
        return parent1 + (parent2 - parent1) * np.random.rand(*parent1.shape)


    def averageCrossover(self):
        """
        Each offspring's genes will be defined as the average between the 
        parents alleles.
        This crossover will create new genes during the iterations.
        Generates one offspring from each pair of parents.
        """

        select = self.selection(2*self.crossoverSize).reshape(2, -1)
        parent1, parent2 = self.population[select]

        return (parent1 + parent2) / 2


    def noCrossover(self):
        """
        Offsprings will be a clone of the selected parents.
        """

        return self.population[self.selection(self.crossoverSize)]


    """
    Mutation methods (one of them must be chosen).
    """

    def uniformMutation(self, population):
        """
        Each selected gene will be changed to a new random value.
        """

        mask, genePositions = self.mutationBy(population.shape)

        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        population[mask] = geneMin + (geneMax - geneMin) \
                                     * np.random.rand(genePositions.size)

        return population


    # TODO: Fix clip limit
    def creepMutation(self, population):
        """
        Each selected gene will be increased or decreased by a random value 
        proportional to 'creepFactor.
        """

        try:
            creepFactor = self.parameters['creepFactor']
        except KeyError:
            creepFactor = self.parameters['creepFactor'] = 0.001

        mask, genePositions = self.mutationBy(population.shape)

        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        geneRange = 2*creepFactor*(geneMax - geneMin)

        population[mask] = (population[mask] + 
            geneRange * (np.random.rand(genePositions.size) - 0.5)
        ).clip(geneMin, geneMax)

        return population


    # TODO: Fix clip limit
    # TODO: Unit gaussian?
    def gaussianMutation(self, population):
        """
        Eache gene will be changed to a new value given by a normal distribution
        with mean equals the old gene value and standard deviation equals
        'gaussianScale'.
        """

        try:
            gaussianScale = self.parameters['gaussianScale']
        except KeyError:
            gaussianScale = self.parameters['gaussianScale'] = 1.0

        mask, genePositions = self.mutationBy(population.shape)

        geneMin = self.lowerBound[genePositions]
        geneMax = self.upperBound[genePositions]

        population[mask] = (population[mask] + 
            np.random.normal(scale = gaussianScale, size = genePositions.size)
        ).clip(geneMin, geneMax)

        return population


    """
    Population and Generations related methods.
    """

    # Generates the first population.
    def populate(self):
        delta = (self.upperBound - self.lowerBound)

        self.population = self.lowerBound + (
            delta * np.random.rand(self.populationSize, self.geneSize)
        ).astype(self.dtype)

        self.values = self.fitness(self.population.T, **self.fArgs)
        self.argSort = self.values.argsort()


    # Generates the next generation's population.
    def nextGeneration(self):
        offspring = self.mutation(self.crossover())

        nonElite = self.argSort[self.eliteSize:]
        self.population[nonElite] = offspring
        self.values[nonElite] = self.fitness(offspring.T, **self.fArgs)

        self.argSort = self.values.argsort()
   

    """
    GA's execution.
    """

    def run(self):
        """
        Runs the Generic Algorithm.
        """
        self.populate()

        for _ in range(self.maxGenerations):
            if self.values[self.argSort[0]] <= self.threshold:
                break

            self.nextGeneration()

        return self.population[self.argSort[0]], self.values[self.argSort[0]]

    def plot(self):
        """
        Plot GA's evolution over the generations.
        """

        import matplotlib.pyplot as plt

        values = np.zeros(self.maxGenerations)
        self.populate()

        for i in range(self.maxGenerations):
            self.nextGeneration()
            values[i] = self.values[self.argSort[0]]

        plt.plot(np.arange(self.maxGenerations), values)
        plt.show()


def GA(fitness,
       size,
       lowerBound=np.NINF,
       upperBound=np.inf,
       mutation='gaussian',
       selection='tournament',
       crossover='uniform',
       dtype=np.float64,
       **kwargs):
    """
    Simple caller for GeneticAlgorithm.
    """

    GA = GeneticAlgorithm(
        # Function to be optimized
        fitness=fitness,
        size=size,

        # Limits for the genes
        lowerBound=lowerBound,
        upperBound=upperBound,

        # Genetic Algorithm methods
        selection=selection,
        mutation=mutation,
        crossover=crossover,

        # Data type
        dtype=dtype,

        # Other parameters
        **kwargs
    )

    return GA.run()
