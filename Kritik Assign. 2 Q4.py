import numpy as np

def roots(f, a, b):

    if f(a) * f(b) >= 0:
        return None

    while True:
        midpoint = (a + b) / 2
        if abs(f(midpoint)) < 1e-10:
            return midpoint
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

def func1(x):
    return np.exp(x) + np.log(x)

def func2(x):
    return np.arctan(x) - x**2

def func3(x):
    return np.sin(x) - np.log(x)

def func4(x):
    return np.log(np.cos(x))

test_cases = [(func1, 0.1, 1),(func2, 0, 2),(func3, 3, 4),(func4, 5, 7)]

for f, a, b in test_cases:
    root = roots(f, a, b)
    if root is not None:
        print(f"Root of f(x) on [{a}, {b}]: {root:.10f}")
    else:
        print(f"No root found for f(x) on [{a}, {b}].")