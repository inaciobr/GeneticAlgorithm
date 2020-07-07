"""
A few functions that can be used to test the optimization algorithms.
"""

import math
import numpy as np


__all__ = ['ackley', 'sphere', 'rastrigin']


def ackley(x):
    """
    Ackley Function
    It has it's global minimum at the point where x[i] = 0.
    """
    xLenght = len(x)
    constTerm = 20 + math.e
    firstExp = -20*np.exp(-0.2*np.sqrt(np.add.reduce(x*x)/xLenght))
    secondExp = -np.exp(np.add.reduce(np.cos(math.tau*x))/xLenght)

    return firstExp + secondExp + constTerm


def sphere(x):
    """
    Sphere Function
    It has it's global minimum at the point where x[i] = 0.
    """
    return np.add.reduce(x*x)


def rastrigin(x):
    """
    Rastrigin Function
    It has it's global minimum at the point where x[i] = 0.
    """
    return 10*len(x) + np.add.reduce(x*x - 10*np.cos(math.tau*x))
