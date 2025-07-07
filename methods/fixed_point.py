# methods/fixed_point.py
import math

def evaluate(gx, x):
    return eval(gx, {"x": x, "math": math})

def fixed_point(gx, x0, tol, max_iter):
    for _ in range(max_iter):
        x1 = evaluate(gx, x0)
        if x1 is None:
            return "Invalid g(x)"
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return "Method failed after max iterations."
