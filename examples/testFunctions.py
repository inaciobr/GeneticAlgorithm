"""
A few functions that can be used to test the optimization algorithms.
"""

import math
import numpy as np


# Ackley Function has it's global minimum at the point where x[i] = 0.
def ackley(x):
	return 20*(1 - np.exp(-0.2*np.sqrt(np.add.reduce(x*x) / x.shape[0]))) - \
		   np.exp(np.add.reduce(np.cos(math.tau*x)) / x.shape[0]) + math.e


# Sphere Function has it's global minimum at the point where x[i] = 0.
def sphere(x):
	return np.add.reduce(x*x)


# Rastrigin Function has it's global minimum at the point where x[i] = 0.
def rastrigin(x):
	return 10*x.shape[0] + np.add.reduce(x*x - 10*np.cos(math.tau*x))
