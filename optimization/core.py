import numpy as np

from .genetic_algorithm import GeneticAlgorithm


__all__ = ['GA']


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

    geneticAlgorithm = GeneticAlgorithm(
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

    return geneticAlgorithm.run()
