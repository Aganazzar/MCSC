# Interpolation Testing

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def f(x):
    return np.exp(x)**2+x

x_samples=np.linspace(0,2,4)
y_samples= f(x_samples)
poly_fit= Polynomial.fit(x_samples, y_samples, deg=3)

poly=poly_fit.convert()
roots=poly.roots()
real_roots= roots[np.isreal(roots)].real

print("Estimated root=", real_roots)

print("Interpolated Polynomial Coefficients:", poly.coef)

x_vals= np.linspace(0,10)
plt.plot(x_vals, f(x_vals))
plt.plot(x_vals, poly(x_vals))

plt.show()