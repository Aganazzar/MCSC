
# Newton-Raphson Method

import sympy as sp
import numpy as np

# Define the symbol and function
x = sp.symbols('x')
f = sp.exp(x)**2+x
f_prime = sp.diff(f, x)

# Convert to numerical functions
f_func = sp.lambdify(x, f, 'numpy')
f_prime_func = sp.lambdify(x, f_prime, 'numpy')

# Newton-Raphson method with iteration count
def newton_raphson(x0, tol=1e-6, max_iter=1000):
    xn = x0
    for n in range(1, max_iter + 1):
        fxn = f_func(xn)
        fpxn = f_prime_func(xn)
        if fpxn == 0:
            raise ZeroDivisionError("Zero derivative. No solution found.")
        x_next = xn - fxn / fpxn
        if abs(x_next - xn) < tol:
            return x_next, n  # ✅ return root and number of iterations
        xn = x_next
    raise ValueError("Exceeded maximum iterations")

# Example usage
initial_guess = 2.3044
root, iterations = newton_raphson(initial_guess)
print(f"Root found: {root}")
print(f"Iterations needed: {iterations}")
