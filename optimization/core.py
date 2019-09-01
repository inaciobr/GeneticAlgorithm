"""
Callers for the optimization algorithms.
"""

from .geneticAlgorithm import GeneticAlgorithm

import math
import numpy as np
import time

__all__ = ['GA', 'GeneticAlgorithm']


# Runs the Genetic Algorithms.
def GA(fitness, size, lowerBound, upperBound,
       selection = 'tournament', mutation = 'gene', crossover = 'uniform',
       **kwargs):

    GA = GeneticAlgorithm(fitness = fitness,
                          size = size,
                          lowerBound = lowerBound,
                          upperBound = upperBound,

                          selection = selection,
                          mutation = mutation,
                          crossover = crossover,
                          **kwargs)
    return GA.run()
