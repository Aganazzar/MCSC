# methods/newton.py
import math

def evaluate(expr, x):
    return eval(expr, {"x": x, "math": math})

def newton(fx, dfx, x0, tol, max_iter):
    for _ in range(max_iter):
        f_val = evaluate(fx, x0)
        df_val = evaluate(dfx, x0)
        if df_val == 0:
            return "Derivative is zero!"
        x1 = x0 - f_val / df_val
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return "Method failed after max iterations."
