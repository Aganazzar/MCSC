# methods/bisection.py
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(fx, x):
    return eval(fx, {"x": x, "math": math})

def bisection(fx, a, b, tol=1e-5, max_iter=100):
    # Check initial condition
    if evaluate(fx, a) * evaluate(fx, b) > 0:
        return None, None, None  # Will handle in main.py as error

    iteration_list = []

    for i in range(1, max_iter + 1):
        mid = (a + b) / 2.0
        f_mid = evaluate(fx, mid)
        if f_mid is None:
            return None, None, None

        iteration_list.append({
            "Iteration": i,
            "a": a,
            "b": b,
            "c": mid,
            "f(c)": f_mid,
            "|b - a|": abs(b - a)
        })

        if abs(f_mid) < tol or (b - a) / 2 < tol:
            root = mid
            break

        if evaluate(fx, a) * f_mid < 0:
            b = mid
        else:
            a = mid
    else:
        root = (a + b) / 2.0

    # Convert to DataFrame
    iteration_data = pd.DataFrame(iteration_list)

    # Create graph
    x_vals = np.linspace(a - 1, b + 1, 400)
    y_vals = [evaluate(fx, x) for x in x_vals]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(root, color='red', linestyle='--', label=f"Root ≈ {root:.5f}")
    ax.set_title("Function Graph")
    ax.grid(True)
    ax.legend()

    return root, iteration_data, fig
