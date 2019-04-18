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

    GA = GeneticAlgorithm(minFunction = minFunction,
                          inputSize = inputSize,
                          lowerBound = lowerBound,
                          upperBound = upperBound,

                          maxIteractions = 100*inputSize if maxIteractions == 0 else maxIteractions,
                          populationSize = min(200, 20*inputSize) if populationSize == 0 else populationSize,
                          eliteNum = math.ceil(populationSize * elitePercentage),
                          threshold = threshold,

                          selectionMethod = selectionMethod,
                          mutationMethod = mutationMethod,
                          crossoverMethod = crossoverMethod,

                          chromosomeMutationRate = chromosomeMutationRate,
                          geneMutationRate = geneMutationRate,
                          tournamentSize = max(2, math.ceil(populationSize * tournamentPercentage)))
    return GA.run()
