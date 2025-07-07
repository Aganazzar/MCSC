# methods/newton.py
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate(expr, x):
    try:
        return eval(expr, {"x": x, "math": math})
    except Exception:
        return None

def newton(fx, dfx, x0, tol=1e-5, max_iter=100):
    iterations = []

    for i in range(1, max_iter + 1):
        f_val = evaluate(fx, x0)
        df_val = evaluate(dfx, x0)

        if f_val is None or df_val is None:
            return None, None, None

        if df_val == 0:
            return None, None, None  # derivative zero error

        x1 = x0 - f_val / df_val

        iterations.append({
            "Iteration": i,
            "x0": x0,
            "f(x0)": f_val,
            "f'(x0)": df_val,
            "x1": x1,
            "|x1 - x0|": abs(x1 - x0)
        })

        if abs(x1 - x0) < tol:
            root = x1
            break

        x0 = x1
    else:
        root = x0

    iteration_data = pd.DataFrame(iterations)

    # Graph f(x)
    x_vals = np.linspace(root - 5, root + 5, 400)
    y_vals = [evaluate(fx, x) if evaluate(fx, x) is not None else float('nan') for x in x_vals]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(root, color='red', linestyle='--', label=f"Root ≈ {root:.5f}")
    ax.set_title("Newton-Raphson Function Graph")
    ax.grid(True)
    ax.legend()

    return root, iteration_data, fig
