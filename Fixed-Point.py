# Fixed point Iteration Method

import numpy as np

# Define the function g(x) such that x = g(x)
def g(x):
    return np.cos(x)  # Example: x = cos(x) has a fixed point near 0.739

# Fixed Point Iteration Method
def fixed_point_iteration(x0, tol=1e-6, max_iter=100):
    for n in range(1, max_iter + 1):
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, n  # ✅ Return fixed point and number of iterations
        x0 = x1
    raise ValueError("Exceeded maximum iterations")

# Example usage
initial_guess = 0.629251135
root, iterations = fixed_point_iteration(initial_guess)
print(f"Fixed point found: {root}")
print(f"Iterations needed: {iterations}")
