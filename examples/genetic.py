import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optimization as opt
import testFunctions

import time
import numpy as np
import cProfile


def minimize():
    nVars = 100                      # Number of variables in the problem.
    func = testFunctions.sphere     # Function to be optimized.

    duration = -time.time()
    minArgs = opt.GA(minFunction = func,
                     inputSize = nVars,
                     lowerBound = np.array([-30.0] * nVars),
                     upperBound = np.array([+30.0] * nVars),

                     maxIteractions = 100 * nVars,
                     populationSize = min(20 * nVars, 200),
                     elitePercentage = 0.05,
                     threshold = np.NINF,

                     selectionMethod = opt.GeneticAlgorithm.tournamentSelect,
                     mutationMethod = opt.GeneticAlgorithm.geneMutation,
                     crossoverMethod = opt.GeneticAlgorithm.uniformCrossover,
                     
                     chromosomeMutationRate = 0.2,
                     geneMutationRate = 0.01,
                     tournamentPercentage = 0.05)
    duration += time.time()

    #print("{:3d}, {:7.4f}, {:5.3f}".format(nVars, duration, func(minArgs)))

    print("Tempo de execução:", duration, "s.\n")
    #print(nVars, "argumentos que minimizam a função:\n", minArgs)
    print("Valor da função neste ponto:", func(minArgs))


if __name__ == "__main__":
    print("===== Genetic Algorithm =====")
    #cProfile.run("minimize()")
    minimize()
