# Regular Farsi Method

import numpy as np

# Define the function whose root you want to find
def f(x):
    return np.cos(x) * np.exp(x)  # Example function

# False Position Method
def false_position(a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("Function has the same sign at a and b. Choose different bounds.")
    
    for n in range(1, max_iter + 1):
        # Compute the point of intersection (Regula Falsi formula)
        c = b - (f(b) * (b - a)) / (f(b) - f(a))
        fc = f(c)

        # Check for convergence
        if abs(fc) < tol:
            return c, n

        # Update interval
        if f(a) * fc < 0:
            b = c
        else:
            a = c

    raise ValueError("Exceeded maximum iterations")

# Example usage
a = 0
b = 1
root, iterations = false_position(a, b)
print(f"Root found: {root}")
print(f"Iterations needed: {iterations}")
