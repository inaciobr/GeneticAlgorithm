"""
Callers for the optimization algorithms.
"""

from .geneticAlgorithm import GeneticAlgorithm

import math
import numpy as np
import time

__all__ = ['GA', 'GeneticAlgorithm']


# Runs the Genetic Algorithms.
def GA(fitness, size, lowerBound, upperBound, maxGenerations = None,
       threshold = np.NINF, populationSize = 200, eliteSize = 10, mutationRate = 0.01,
       selection = 'tournament', mutation = 'gene', crossover = 'uniform', mutationBy = 'gene',
       **kwargs):

    GA = GeneticAlgorithm(fitness = fitness,
                          size = size,
                          lowerBound = lowerBound,
                          upperBound = upperBound,

                          maxGenerations = maxGenerations,
                          threshold = threshold,

                          populationSize = populationSize,
                          eliteSize = eliteSize,

                          selection = selection,
                          mutation = mutation,
                          crossover = crossover,

                          mutationBy = mutationBy,
                          mutationRate = mutationRate,
                          **kwargs)
    return GA.run()
