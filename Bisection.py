# Bisection Method

import numpy as np

def f(x):
    return np.cos(x) * np.exp(x)  # Example function

def bisection(a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("Function must have opposite signs at a and b.")

    for n in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol or (b - a)/2 < tol:
            return c, n

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    raise ValueError("Exceeded maximum iterations")

# Example usage
a, b = 0, 1
root, iterations = bisection(a, b)
print(f"Bisection Root: {root}")
print(f"Iterations: {iterations}")
