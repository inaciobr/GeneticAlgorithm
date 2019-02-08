# Optimization
Optimization algorithms included so far:
+ Genetic Algorithm

## 1. Genetic Algorithm
Example of usage:

```python
from optimization import opt

# Returns the arguments that minimize the function
minArgs = opt.GA(function = func,                           # function to be minimized
                 nArgs = nArgs,                             # Number of arguments in the function
                 lowerBound = np.array([-30.0] * nArgs),    # Lower bound of each argument
                 upperBound = np.array([+30.0] * nArgs),    # Upper bound of each argument
                 threshold = np.NINF,                       # Threshold to stop the algorithm

                 maxIteractions = 100 * nArgs,              # Number of generations. The default is 100 * nArgs
                 populationSize = min(20 * nArgs, 200),     # Number of individuals in each generation. Usually it is limited to 200
                 elitePercent = 0.05,                       # Percentage of elite individuals. The default is 5%

                 selectionMethod = opt.geneticAlgorithm.tournamentSelect,   # Selection method
                 tournamentSize = 0.05,                                     # Size of the tournament in percentage of the population (if used)

                 mutationMethod = opt.geneticAlgorithm.geneMutation,        # Mutation method
                 geneMutationRate = 0.01,                                   # Gene mutation rate (if used)
                 chromosomeMutationRate = 0.2)                              # Chromosome mutation rate (if used)
```

It's possible to run an example using the following command:

    python -m Optimization.examples.genetic

*Note*: The parameters should be adjusted to every specific problem, but the configuration shown in the example can be a good starting point.
Some parameters like the _mutation rates_ and the _tournamentSize_ have an optimal value where the algorithm will be able to find a better solution and the value can be adjusted to maximize the efficience of the algorithm to the problem. *Changing these values won't considerably change the speed of the algorithm*.
On the other hand, parameters like _maxIteractions_ and _populationSize_ have a big impact on the speed of the algorithm *AND* on the quality of the solution, so they have to be configurated in a way that the program can provide a solution good enough in the time required. The combination of the two arguments mentioned also depends on the problem, but in general, more iterations provide a better solution than bigger populations, so, by default, the population is limited to 200 individuals.

### 1.1 Selection Methods

### 1.2 Mutation Methods

