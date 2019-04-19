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
On the other hand, parameters like _maxIteractions_ and _populationSize_ have a big impact on the speed of the algorithm *AND* on the quality of the solution, so they have to be configurated in a way that the program can provide a solution good enough and in the time required.
The combination of the two arguments mentioned also depends on the problem, but in general, more iterations provide a better solution than bigger populations, so it's usual to limit the size of the population.

It is also important to notice that in some problems, the choice of the right methods may be relevant. In this case, it's possible to choose different methods for selection, mutation and crossover. The module geneticAlgorithm provides some methods, but it also allows the developer to create new ones.
The methods can be selected by passing them in the arguments of the GA function:

### 1.1 Selection Methods
* Roulette Wheel selection
```python
selectionMethod = opt.GeneticAlgorithm.wheelSelect
```

* Tournament selection
```python
selectionMethod = opt.GeneticAlgorithm.tournamentSelect
```

* Stochastic selection
```python
selectionMethod = opt.GeneticAlgorithm.stochasticSelect
```


### 1.2 Mutation Methods
* Chromosome mutation
```python
mutationMethod = opt.GeneticAlgorithm.chromosomeMutation
```

* Gene mutation
```python
mutationMethod = opt.GeneticAlgorithm.geneMutation
```

### 1.3 Crossover Methods
* Uniform crossover
```python
crossoverMethod = opt.GeneticAlgorithm.uniformCrossover
```

* Single Point crossover
```python
crossoverMethod = opt.GeneticAlgorithm.singlePointCrossover
```

* Two Point crossover
```python
crossoverMethod = opt.GeneticAlgorithm.twoPointCrossover
```

* No crossover
```python
crossoverMethod = opt.GeneticAlgorithm.noCrossover
```
