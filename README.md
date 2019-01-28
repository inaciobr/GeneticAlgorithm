# Optimization
Optimization algorithms included:
- Genetic Algorithm

## 1. Genetic Algorithm

```python
from optimization import opt

# Returns the arguments that minimize the function
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
```

It's possible to run the example using the following command:
    python -m Optimization.examples.genetic
