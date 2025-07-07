# methods/fixed_point.py
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate(gx, x):
    try:
        return eval(gx, {"x": x, "math": math})
    except Exception:
        return float('nan')

def fixed_point(gx, x0, tol=1e-5, max_iter=100):
    iterations = []

    for i in range(1, max_iter + 1):
        x1 = evaluate(gx, x0)
        if math.isnan(x1):
            return None, None, None

        iterations.append({
            "Iteration": i,
            "x0": x0,
            "x1 = g(x0)": x1,
            "|x1 - x0|": abs(x1 - x0)
        })

        if abs(x1 - x0) < tol:
            root = x1
            break

        x0 = x1
    else:
        root = x0

    iteration_data = pd.DataFrame(iterations)

    # Plot f(x) = g(x) - x safely
    x_vals = np.linspace(root - 5, root + 5, 400)
    y_vals = []
    for x in x_vals:
        val = evaluate(gx, x)
        if math.isnan(val):
            y_vals.append(float('nan'))
        else:
            y_vals.append(val - x)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x) = g(x) - x")
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(root, color='red', linestyle='--', label=f"Root ≈ {root:.5f}")
    ax.set_title("Fixed-Point Function Graph")
    ax.grid(True)
    ax.legend()

    return root, iteration_data, fig
