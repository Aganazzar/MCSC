# methods/secant.py
import math

def evaluate(fx, x):
    return eval(fx, {"x": x, "math": math})

def secant(fx, x0, x1, tol, max_iter):
    for _ in range(max_iter):
        f0 = evaluate(fx, x0)
        f1 = evaluate(fx, x1)
        if f1 - f0 == 0:
            return "Division by zero"
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return "Method failed after max iterations."
