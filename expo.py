import numpy as np
import sympy as sp
from sympy import symbols, diff, simplify, solve, Eq, lambdify

# Define symbolic variable
x = symbols('x')

def interpolate_and_find_best_root(func_expr_str):
    # Step 1: Convert string to symbolic expression
    func_expr = sp.sympify(func_expr_str, locals={'e': sp.E})

    # Step 2: Lambdify for numerical evaluation
    f = sp.lambdify(x, func_expr, modules=["numpy"])
    
    # Step 3: Generate interpolation points
    x_vals = np.linspace(-2, 2, 4)
    y_vals = f(x_vals)

    # Step 4: Build Lagrange interpolating polynomial
    P = 0
    for i in range(4):
        xi = x_vals[i]
        yi = y_vals[i]
        Li = 1
        for j in range(4):
            if i != j:
                xj = x_vals[j]
                Li *= (x - xj) / (xi - xj)
        P += yi * Li

    P = simplify(P)
    print(f"[INFO] Cubic Interpolation Polynomial:\nP(x) = {P}\n")

    # Step 5: Solve for real roots
    roots = solve(Eq(P, 0), x)
    roots = [r.evalf() for r in roots if r.is_real]

    if not roots:
        print("[WARNING] No real roots found in the interpolated polynomial.")
        return

    # Step 6: Compute slope of original function at each root
    f_prime = diff(func_expr, x)
    slopes = []
    for r in roots:
        slope = abs(f_prime.subs(x, r).evalf())
        print(f"Root: {r}, |f'(x)|: {slope}")
        slopes.append((r, slope))

    # Step 7: Select root with highest slope
    best_root, best_slope = max(slopes, key=lambda t: t[1])
    print(f"\n[RESULT] Root with maximum slope: x = {best_root}, |f'(x)| = {best_slope}")

# === Example Usage ===
expr = "exp(x) + x^2 + 10*x"
interpolate_and_find_best_root(expr)
