"""
Callers for the optimization algorithms.
"""

from .geneticAlgorithm import GeneticAlgorithm

import math
import numpy as np
import time

__all__ = ['GA', 'GeneticAlgorithm']


# Runs the Genetic Algorithms.
def GA(minFunction, inputSize, lowerBound, upperBound, maxIteractions = 0, populationSize = 0, elitePercentage = 0.05,
       threshold = np.NINF, selectionMethod = None, mutationMethod = None, crossoverMethod = None,
       chromosomeMutationRate = 0.2, geneMutationRate = 0.01, tournamentPercentage = 0.1):

    GA = GeneticAlgorithm(fitnessFunction = minFunction,
                          geneSize = inputSize,
                          lowerBound = lowerBound,
                          upperBound = upperBound,

                          maxIteractions = maxIteractions if maxIteractions else 100*inputSize,
                          populationSize = populationSize if populationSize else min(200, 20*inputSize),
                          eliteSize = math.ceil(populationSize * elitePercentage),
                          threshold = threshold,

                          selectionMethod = selectionMethod,
                          mutationMethod = mutationMethod,
                          crossoverMethod = crossoverMethod,

                          chromosomeMutationRate = chromosomeMutationRate,
                          geneMutationRate = geneMutationRate,
                          tournamentSize = max(2, math.ceil(populationSize * tournamentPercentage)))
    return GA.run()
