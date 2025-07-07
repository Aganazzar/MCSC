# methods/bisection.py
import math

def evaluate(fx, x):
    return eval(fx, {"x": x, "math": math})

def bisection(fx, a, b, tol, max_iter):
    if evaluate(fx, a) * evaluate(fx, b) > 0:
        return "f(a) and f(b) must have opposite signs!"
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        f_mid = evaluate(fx, mid)
        if f_mid is None:
            return "Invalid function"
        if abs(f_mid) < tol or (b - a) / 2 < tol:
            return mid
        if evaluate(fx, a) * f_mid < 0:
            b = mid
        else:
            a = mid
    return "Method failed after max iterations."

