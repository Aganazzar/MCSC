# Curve Fitting(Obsolete):
import numpy as np

from scipy.optimize import curve_fit

def f(x):
    return np.exp(x)**2+x

# Define a model: e.g., quadratic or cubic
def model(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

# Sample the function
x_data = np.linspace(1, 2, 10)
y_data = f(x_data)

# Fit the curve
params, _ = curve_fit(model, x_data, y_data)
a, b, c, d = params

# Create a polynomial and find roots
coeffs = [a, b, c, d]
roots = np.roots(coeffs)
real_roots = roots[np.isreal(roots)].real

print("Estimated root(s) from curve fitting:", real_roots)
