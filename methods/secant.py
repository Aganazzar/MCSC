# methods/secant.py
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate(fx, x):
    try:
        return eval(fx, {"x": x, "math": math})
    except Exception:
        return None

def secant(fx, x0, x1, tol=1e-5, max_iter=100):
    iterations = []

    for i in range(1, max_iter + 1):
        f0 = evaluate(fx, x0)
        f1 = evaluate(fx, x1)

        if f0 is None or f1 is None:
            return None, None, None

        denominator = f1 - f0
        if denominator == 0:
            return None, None, None  # division by zero

        x2 = x1 - f1 * (x1 - x0) / denominator

        iterations.append({
            "Iteration": i,
            "x0": x0,
            "x1": x1,
            "f(x0)": f0,
            "f(x1)": f1,
            "x2": x2,
            "|x2 - x1|": abs(x2 - x1)
        })

        if abs(x2 - x1) < tol:
            root = x2
            break

        x0, x1 = x1, x2
    else:
        root = x1

    iteration_data = pd.DataFrame(iterations)

    # Graph f(x)
    x_vals = np.linspace(root - 5, root + 5, 400)
    y_vals = [evaluate(fx, x) if evaluate(fx, x) is not None else float('nan') for x in x_vals]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(root, color='red', linestyle='--', label=f"Root ≈ {root:.5f}")
    ax.set_title("Secant Function Graph")
    ax.grid(True)
    ax.legend()

    return root, iteration_data, fig
