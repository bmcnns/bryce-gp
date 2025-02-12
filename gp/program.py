import random

import numpy as np

def add(x, y):
    return np.add(x, y)

def sub(x, y):
    return np.subtract(x, y)

def mul(x, y):
    return np.multiply(x, y)

def div(x, y):
    return np.divide(x, y, where=(y != 0), out=np.ones_like(x))

def sin(x, y):
    return np.sin(x)

def power(x, y):
    # Temporarily set NumPy to raise an error on overflow.
    with np.errstate(over='raise'):
        try:
            result = np.power(x, y)
        except FloatingPointError:
            return 0
    # Even if no exception was raised, check for infinite results.
    if np.isscalar(result) and (result == np.inf or result == -np.inf):
        return 0
    return result

class Program:
    def __init__(self, num_features, num_outputs, depth=3):
        self.depth = depth
        self.input_size = num_features
        self.output_size = num_outputs

        if self.is_leaf_():
            if random.choice([True, False]):
                self.value = random.uniform(-10, 10)
                self.index = None
            else:
                self.value = None
                self.index = random.randrange(0, num_features)
            self.operator = None
            self.left = None
            self.right = None
        else:
            self.value = None
            self.index = None
            self.operator = random.choice([add, sub, mul, div])
            self.left = Program(self.input_size, self.output_size, random.randint(0, depth - 1))
            self.right = Program(self.input_size, self.output_size, random.randint(0, depth - 1))

    def is_leaf_(self):
        return self.depth == 0

    def is_observation_(self):
        return self.is_leaf_() and self.index is not None

    def is_constant_(self):
        return self.is_leaf_() and self.index is None

    def size_(self):
        if self.is_leaf_():
            return 1
        left_size = self.left.size_() if self.left else 0
        right_size = self.right.size_() if self.right else 0
        return 1 + left_size + right_size

    def complexity(self):
        total = self.size_()
        if self.left:
            total += self.left.complexity()
        if self.right:
            total += self.right.complexity()
        return total

    def __repr__(self):
        if self.is_leaf_() and self.is_observation_():
            return f"X{self.index}"
        elif self.is_leaf_() and self.is_constant_():
            return str(f"{self.value:.2f}")
        elif self.left and self.right:
            return f"({self.operator.__name__} {self.left} {self.right})"

    def __call__(self, X):
        X = np.asarray(X).astype(np.float64)

        if self.is_constant_():
            return self.value * np.ones(shape=(X.shape[0]), dtype=np.float64)
        elif self.is_observation_():
            return X[:, self.index]
        else:
            left_value = self.left(X)
            right_value = self.right(X)
            return self.operator(left_value, right_value)