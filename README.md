# Optimization
Optimization algorithms included so far:
1. Genetic Algorithm

## 1. Genetic Algorithm
Example of usage:

```python
import optimization as opt

# Returns the arguments that minimize the function
minArgs = opt.GA(minFunction = func, # Function to be minimized
                inputSize = nVars, # Number of arguments in the function
                lowerBound = np.array([-30.0] * nVars), # Lower bound of each argument
                upperBound = np.array([+30.0] * nVars), # Upper bound of each argument

                maxIteractions = 100 * nVars, # Number of generations. The default value is 100 * nArgs
                populationSize = min(20 * nVars, 200), # Number of individuals in each generation. Usually it is limited to 200
                elitePercentage = 0.05, # Percentage of elite individuals. The default is 5%
                threshold = np.NINF, # Threshold to stop the algorithm

                selectionMethod = opt.GeneticAlgorithm.tournamentSelect, # Selection method
                mutationMethod = opt.GeneticAlgorithm.geneMutation, # Mutation method
                crossoverMethod = opt.GeneticAlgorithm.uniformCrossover, # Crossover method
                
                chromosomeMutationRate = 0.2, # Chromosome mutation rate (if used)
                geneMutationRate = 0.01, # Gene mutation rate (if used)
                tournamentPercentage = 0.05) # Percentage of the population in a tournament (if used)
```

*Note*: The parameters should be adjusted to every specific problem, but the configuration shown in the example can be a good starting point.
Some parameters like the _geneMutationRate_ and the _tournamentPercentage_ have an optimal value where the algorithm will be able to find a better solution and the value can be adjusted to maximize the efficience of the algorithm to the problem. *Changing these values won't considerably change the speed of the algorithm*.
<br />On the other hand, parameters like _maxIteractions_ and _populationSize_ have a big impact on the speed of the algorithm *AND* on the quality of the solution, so they have to be configurated in a way that the program can provide a solution good enough and in the time required.
<br />The combination of the two arguments mentioned also depends on the problem, but in general, more iterations provide a better solution than bigger populations, so it's usual to limit the size of the population.

### 1.1 Selection Methods

### 1.2 Mutation Methods

### 1.3 Crossover Methods

