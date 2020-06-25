from optimization.geneticAlgorithm import GA
from optimization.examples import test_functions

import time
import numpy as np
# import cProfile


def minimize():
    nDims = 100
    objFunction = test_functions.sphere

    duration = -time.time()

    minArgs, val = GA(
        fitness=objFunction,
        size=nDims,

        lowerBound=-30.0,
        upperBound=+30.0,

        mutation='gaussian',
        selection='tournament',
        crossover='singlePoint',

        mutationBy='chromosome',
        dtype=np.float64,
    )

    duration += time.time()

    print("Tempo de execução: {}s.".format(duration))
    # print("{} argumentos que minimizam a função: {}".format(nDims, minArgs))
    print("Valor da função neste ponto: {}".format(val))


if __name__ == "__main__":
    print("===== Genetic Algorithm =====")
    # cProfile.run("minimize()")
    minimize()
