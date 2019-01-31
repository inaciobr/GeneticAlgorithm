from ..optimization import opt
from .testFunctions import *

import time
import numpy as np

def minimize():
    nArgs = 50      # Number of variables in the problem.
    func = ackley   # Function to be optimized

    duration = -time.time()
    minArgs = opt.GA(function = func,
                     nArgs = nArgs,
                     lowerBound = np.array([-30.0] * nArgs),
                     upperBound = np.array([+30.0] * nArgs),
                     threshold = np.NINF,

                     maxIteractions = 100 * nArgs,
                     populationSize = min(20 * nArgs, 200),
                     elitePercent = 0.05,

                     selectionMethod = opt.geneticAlgorithm.tournamentSelect,
                     tournamentSize = 0.05,

                     mutationMethod = opt.geneticAlgorithm.geneMutation,
                     geneMutationRate = 0.01,
                     chromosomeMutationRate = 0.2)
    duration += time.time()

    #print("{:3d}, {:7.4f}, {:5.3f}".format(nArgs, duration, func(minArgs)))

    print("Tempo de execução:", duration, "s.\n")
    print(nArgs, "argumentos que minimizam a função:\n", minArgs)
    print("Valor da função neste ponto:", func(minArgs))


if __name__ == "__main__":
    print("===== Genetic Algorithm =====")
    minimize()
