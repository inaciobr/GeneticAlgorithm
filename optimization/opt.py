"""
Callers for the optimization algorithms.
"""

from .geneticAlgorithm import geneticAlgorithm

import math
import numpy as np

# Runs the Genetic Algorithms.
def GA(function, nArgs, lowerBound, upperBound, maxIteractions = 0, populationSize = 0, threshold = np.NINF,
        elitePercent = 0.05, tournamentSize = 0.1, chromosomeMutationRate = 0.2, geneMutationRate = 0.01,
        selectionMethod = None, mutationMethod = None, crossoverMethod = None):

    GA = geneticAlgorithm(function = function,
                          nArgs = nArgs,
                          lowerBound = lowerBound,
                          upperBound = upperBound,
                          maxIteractions = 100 * nArgs if maxIteractions == 0 else maxIteractions,
                          populationSize = min(20 * nArgs, 200) if populationSize == 0 else populationSize,
                          threshold = threshold,
                          eliteNum = math.ceil(populationSize * elitePercent),
                          chromosomeMutationRate = chromosomeMutationRate,
                          geneMutationRate = geneMutationRate,
                          tournamentSize = max(2, math.ceil(populationSize * tournamentSize)),
                          selectionMethod = selectionMethod if selectionMethod is not None else geneticAlgorithm.tournamentSelect,
                          mutationMethod = mutationMethod if mutationMethod is not None else geneticAlgorithm.geneMutation,
						  crossoverMethod = crossoverMethod if crossoverMethod is not None else geneticAlgorithm.crossoverUniform)
    return GA.run()
