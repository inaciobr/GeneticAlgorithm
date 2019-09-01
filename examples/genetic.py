import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optimization as opt
import testFunctions

import time
import numpy as np
import cProfile


def minimize():
    nVars = 100                     # Number of variables in the problem.
    func = testFunctions.sphere     # Function to be optimized.

    duration = -time.time()
    minArgs = opt.GA(fitness = func,
                     size = nVars,
                     lowerBound = -30.0,
                     upperBound = +30.0,

                     selection = 'tournament',
                     mutation = 'gaussian',
                     crossover = 'uniform',

                     mutationBy = 'chromosome',
                     geneMutationRate = 0.75,
                     tournamentSize = 10)
    duration += time.time()

    #print("{:3d}, {:7.4f}, {:5.3f}".format(nVars, duration, func(minArgs)))

    print("Tempo de execução:", duration, "s.\n")
    #print(nVars, "argumentos que minimizam a função:\n", minArgs)
    print("Valor da função neste ponto:", func(minArgs))


if __name__ == "__main__":
    #print("===== Genetic Algorithm =====")
    #cProfile.run("minimize()")
    minimize()
